#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <queue>
#include <string.h>

#if defined(__AVX2__) ||                                                       \
    defined(__FMA__) // icpc, gcc and clang register __FMA__, VS does not
#define VECTORIZE 1
#define AVX_LOOP _Pragma("omp simd")
#else
#define AVX_LOOP // empty
#endif

// enum of settings for building hierarchical softmax tree. do not modify.
#define HSM_INIT_DEGREE 0
#define HSM_INIT_PR 1
#define HSM_INIT_DW 2

#ifndef UINT64_C // VS can not detect the ##ULL macro
#define UINT64_C(c) (c##ULL)
#endif

#define SIGMOID_BOUND 6.0  // computation range for fast sigmoid lookup table
#define DEFAULT_ALIGN 128  // default align in bytes
#define MAX_CODE_LENGTH 64 // maximum HSM code length. sufficient for nv < int64

using namespace std;

typedef unsigned long long ull;
typedef unsigned char byte;

// <MODEL_DEF>
#ifndef INIT_HSM
#define INIT_HSM                                                               \
  HSM_INIT_PR // change here to use different HSM initialization. PR init tested
// to be the best cost/performance ratio
#define LOWMEM_HSM                                                             \
  0 // change to 1 to use 64 bits for HSM tree construction. Will fail for
// graphs bigger than 500k nodes
#endif
// </MODEL_DEF>

#if !defined(INIT_HSM)
#error USE_HSM must be set
#endif

int verbosity = 2; // verbosity level. 2 = report progress and tell jokes, 1 =
                   // report time and hsm size, 0 = errors, <0 = shut up
int n_threads = 1; // number of threads program will be using
float initial_lr = 0.025f; // initial learning rate
int n_hidden = 128;  // DeepWalk parameter "d" = embedding dimensionality aka
                     // number of nodes in the hidden layer
int dw_n_walks = 80; // DeepWalk parameter "\gamma" = walks per vertex
int dw_walk_length = 80; // DeepWalk parameter "t" = length of the walk
int dw_window_size = 10; // DeepWalk parameter "w" = window size
#if INIT_HSM == HSM_INIT_PR
int num_pr_walks = 100; // Implementation parameter, number of walks per node in
                        // the PageRank initialization
#endif

ull step = 0; // global atomically incremented step counter

ull nv = 0, ne = 0; // number of nodes and edges
                    // We use CSR format for the graph matrix (unweighted).
                    // Adjacent nodes for vertex i are stored in
                    // edges[offsets[i]:offsets[i+1]]
int *offsets;       // CSR index pointers for nodes.
int *edges;         // CSR offsets
int *train_order;   // We shuffle the nodes for better performance

float *wVtx; // Vertex embedding, aka DeepWalk's \Phi
float *wCtx; // Hierarchical Softmax tree

float *hsm_weights; // Weights (probabilities) for constructing HSM tree
#if LOWMEM_HSM
ull *hsm_codes; // HSM codes for each vertex
                // We employ format similar to CSR to store nv*MAX_CODE_LENGTH
                // matrix. It is faster and more memory effecient than default
                // word2vec implementations
#else
byte *hsm_codes; // HSM codes for each vertex
#endif
int *hsm_ptrs;    // HSM pointers for each vertex
int *hsm_indptrs; // HSM offsets for each vertex

const int sigmoid_table_size = 1024; // This should fit in L1 cache
const float SIGMOID_RESOLUTION = sigmoid_table_size / (SIGMOID_BOUND * 2.0f);
float *sigmoid_table;

// http://xoroshiro.di.unimi.it/#shootout
// We use xoroshiro128+, the fastest generator available
uint64_t rng_seed[2];

void init_rng(uint64_t seed) {
  for (int i = 0; i < 2; i++) {
    ull z = seed += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    rng_seed[i] = z ^ (z >> 31);
  }
}

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

uint64_t lrand() {
  const uint64_t s0 = rng_seed[0];
  uint64_t s1 = rng_seed[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  rng_seed[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
  rng_seed[1] = rotl(s1, 36);
  return result;
}

static inline double drand() {
  const union un {
    uint64_t i;
    double d;
  } a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};
  return a.d - 1.0;
}

inline int irand(int max) { return lrand() % max; }

inline int irand(int min, int max) { return lrand() % (max - min) + min; }

inline void *
aligned_malloc(size_t size,
               size_t align) { // universal aligned allocator for win & linux
#ifndef _MSC_VER
  void *result;
  if (posix_memalign(&result, align, size))
    result = 0;
#else
  void *result = _aligned_malloc(size, align);
#endif
  return result;
}

inline void aligned_free(void *ptr) { // universal aligned free for win & linux
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

void init_sigmoid_table() { // this shoould be called before fast_sigmoid once
  sigmoid_table = static_cast<float *>(
      aligned_malloc((sigmoid_table_size + 1) * sizeof(float), DEFAULT_ALIGN));
  for (int k = 0; k != sigmoid_table_size; k++) {
    float x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
    sigmoid_table[k] = 1 / (1 + exp(-x));
  }
}

float fast_sigmoid(float x) {
  if (x > SIGMOID_BOUND)
    return 1;
  if (x < -SIGMOID_BOUND)
    return 0;
  int k = (x + SIGMOID_BOUND) * SIGMOID_RESOLUTION;
  return sigmoid_table[k];
}

inline int sample_neighbor(int node) { // sample neighbor node from a graph
  if (offsets[node] == offsets[node + 1])
    return -1;
  return edges[irand(offsets[node], offsets[node + 1])];
}

#if INIT_HSM == HSM_INIT_PR
void estimate_pr_rw(
    float *outputs, int samples = 1000000,
    float alpha = 0.85) { // fills the first argument with random walk counts
  memset(outputs, 0, nv * sizeof(float));
#pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < samples; i++) {
    int current_node = irand(nv);
    outputs[current_node]++;
    while (drand() < alpha) { // kill with probability 1-alpha
      current_node = sample_neighbor(current_node);
      if (current_node == -1)
        break;
      outputs[current_node]++;
    }
  }
  if (verbosity >= 2)
    cout << "PR estimate complete" << endl;
}
#endif

#if INIT_HSM == HSM_INIT_DW
void estimate_dw_probs(float *outputs) { // fills the first argument with counts
                                         // from the DeepWalk-like random walk
                                         // process. it is slower than
                                         // estimate_pr_rw with no effect on
                                         // performance
  memset(outputs, 0, nv * sizeof(float));
#pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < nv; i++) {
    if (verbosity >= 2 && i % 100000 == 0)
      cout << "." << flush;
    for (int j = 0; j < dw_n_walks; j++) {
      int curnode = i;
      for (int k = 1; k < dw_walk_length; k++) {
        outputs[curnode]++;
        curnode = sample_neighbor(curnode);
        if (curnode == -1)
          break;
      }
      outputs[curnode]++;
    }
  }
  if (verbosity >= 2)
    cout << endl;
}
#endif

void shuffle(int *a, int n) { // shuffles the array a of size n
  for (int i = n - 1; i >= 0; i--) {
    int j = irand(i + 1);
    int temp = a[j];
    a[j] = a[i];
    a[i] = temp;
  }
}

void init_hsm(
    float *probs) { // initializes global arrays of HSM from probs array
  if (verbosity > 0)
    cout << "Constructing HSM tree" << flush;
  vector<size_t> idx(nv); // index array for vertices
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), // we need to sort the index array as in probs
       [&probs](size_t i1, size_t i2) { return probs[i1] > probs[i2]; });
  if (verbosity > 1)
    cout << "." << flush;
  float *count = static_cast<float *>(calloc(nv * 2 + 1, sizeof(float)));
  byte *binary = static_cast<byte *>(calloc(nv * 2 + 1, sizeof(byte)));
  int *parent_node = static_cast<int *>(calloc(nv * 2 + 1, sizeof(int)));
  for (int a = 0; a < nv; a++)
    count[a] = probs[idx[a]];
  for (int a = nv; a < nv * 2; a++)
    count[a] = 1e25;
  int pos1 = nv - 1;
  int pos2 = nv;
  int min1i, min2i; // relentlessly copied from word2vec
  for (int a = 0; a < nv - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[nv + a] = count[min1i] + count[min2i];
    parent_node[min1i] = nv + a;
    parent_node[min2i] = nv + a;
    binary[min2i] = 1;
  }
  if (verbosity > 1)
    cout << "." << flush;
  hsm_indptrs =
      static_cast<int *>(aligned_malloc((nv + 1) * sizeof(int), DEFAULT_ALIGN));
  int total_len = 0;
  for (int a = 0; a < nv; a++) {
    int b = a;
    int i = 0;
    while (true) {
      total_len++;
      i++;
      b = parent_node[b];
      if (b == nv * 2 - 2)
        break;
    }
    hsm_indptrs[idx[a]] = -i;
  }
  hsm_indptrs[nv] = total_len;
  for (int i = nv - 1; i >= 0; i--)
    hsm_indptrs[i] += hsm_indptrs[i + 1];
  hsm_ptrs = static_cast<int *>(
      aligned_malloc((total_len + 1) * sizeof(int), DEFAULT_ALIGN));
#if LOWMEM_HSM
  hsm_codes =
      static_cast<ull *>(aligned_malloc(nv * sizeof(ull), DEFAULT_ALIGN));
  memset(hsm_codes, 0, nv * sizeof(ull));
#else
  hsm_codes = static_cast<byte *>(
      aligned_malloc((total_len + 1) * sizeof(byte), DEFAULT_ALIGN));
  memset(hsm_codes, 0, (total_len + 1) * sizeof(byte));
#endif
  int point[MAX_CODE_LENGTH];
  byte code[MAX_CODE_LENGTH];
  for (int a = 0; a < nv; a++) {
    int b = a;
    int i = 0;
    while (true) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == nv * 2 - 2)
        break;
    }
    int ida = idx[a];
    int curptr = hsm_indptrs[ida];
    for (b = 0; b < i; b++) {
#if LOWMEM_HSM
      hsm_codes[ida] ^= (hsm_codes[ida] ^ -code[b]) & // set bit i - b - 1
                        1 << i - b - 1; // faith in operator priority
#else
      hsm_codes[curptr + i - b - 1] = code[b];
#endif
      hsm_ptrs[curptr + i - b] = point[b] - nv;
    }
  }
  for (int a = 0; a < nv; a++)
    hsm_ptrs[hsm_indptrs[idx[a]]] = nv - 2;
  if (verbosity > 0)
    cout << "." << endl
         << "Done! Average code size: " << hsm_indptrs[nv] / float(nv) << endl
         << flush;
  free(count);
  free(binary);
  free(parent_node);
}

int ArgPos(char *str, int argc, char **argv) {
  for (int a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        cout << "Argument missing for " << str << endl;
        exit(1);
      }
      return a;
    }
  return -1;
}

inline void update( // update the embedding, putting w_t gradient in w_t_cache
    float *w_s, float *w_t, float *w_t_cache, float lr, const int label) {
  float score = 0; // score = dot(w_s, w_t)
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    score += w_s[c] * w_t[c];
  score = (label - fast_sigmoid(score)) * lr;
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t_cache[c] += score * w_s[c]; // w_t gradient
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c]; // w_s gradient
}

void Train() {
  ull total_steps = dw_n_walks * nv;
#pragma omp parallel num_threads(n_threads)
  {
    int tid = omp_get_thread_num();
    const int trnd = irand(nv);
    ull ncount = 0;
    ull local_step = 0;
    float lr = initial_lr;
    int *dw_rw = static_cast<int *>(
        aligned_malloc(dw_walk_length * sizeof(int),
                       DEFAULT_ALIGN)); // we cache one random walk per thread
    float *cache = static_cast<float *>(aligned_malloc(
        n_hidden * sizeof(float),
        DEFAULT_ALIGN)); // cache for updating the gradient of a node
#pragma omp barrier
    while (true) {
      if (ncount > 10) { // update progress every now and then
#pragma omp atomic
        step += ncount;
        if (step > total_steps) // note than we may train for a little longer
                                // than user requested
          break;
        if (tid == 0)
          if (verbosity > 1)
            cout << fixed << setprecision(6) << "\rlr " << lr << ", Progress "
                 << setprecision(2) << step * 100.f / (total_steps + 1) << "%";
        ncount = 0;
        local_step = step;
      }
      lr = initial_lr *
           (1 - step / static_cast<float>(total_steps + 1)); // linear LR decay
      if (lr < initial_lr * 0.0001)
        lr = initial_lr * 0.0001;

      dw_rw[0] =
          train_order[(local_step + ncount + trnd) % nv]; // get shuffled source
                                                          // node. trnd makes
                                                          // sure we train on
                                                          // different parts of
                                                          // the graph
      for (int i = 1; i < dw_walk_length; i++)
        dw_rw[i] =
            sample_neighbor(dw_rw[i - 1]); // sample random walk from the source

      for (int dwi = 0; dwi < dw_walk_length; dwi++) {
        int b = irand(dw_window_size); // subsample window size
        int n1 = dw_rw[dwi];
        if (n1 == -1)
          break;
        for (int dwj = max(0, dwi - dw_window_size + b);
             dwj < min(dwi + dw_window_size - b + 1, dw_walk_length); dwj++) {
          if (dwi == dwj)
            continue;
          int n2 = dw_rw[dwj];
          if (n2 == -1)
            break;
          if (n1 == n2)
            continue;
          memset(cache, 0, n_hidden * sizeof(float)); // clear cache
#if LOWMEM_HSM
          ull code = hsm_codes[n1];
#endif
          for (int hsi = hsm_indptrs[n1]; hsi < hsm_indptrs[n1 + 1]; hsi++) {
            int tou = hsm_ptrs[hsi]; // pointer at level hsi

            int lab =
#if LOWMEM_HSM
                code >> hsi - hsm_indptrs[n1] &
                1; // label at level hsi - hsm_indptrs[n1]
#else
                hsm_codes[hsi];
#endif
            update(&wCtx[tou * n_hidden], &wVtx[n2 * n_hidden], cache, lr, lab);
          }
          AVX_LOOP
          for (int c = 0; c < n_hidden; c++) // update cache of \Phi[n2]
            wVtx[n2 * n_hidden + c] += cache[c];
        }
      }
      ncount++;
    }
  }
}

int main(int argc, char **argv) {
  int a;
  string network_file, embedding_file;
  ull seed = time(nullptr); // default seed is somewhat random
  init_sigmoid_table();
  if ((a = ArgPos(const_cast<char *>("-input"), argc, argv)) > 0)
    network_file = argv[a + 1];
  else {
    if (verbosity > 0)
      cout << "Input file not given! Aborting now.." << endl;
    return 1;
  }
  if ((a = ArgPos(const_cast<char *>("-output"), argc, argv)) > 0)
    embedding_file = argv[a + 1];
  else {
    if (verbosity > 0)
      cout << "Output file not given! Aborting now.." << endl;
    return 1;
  }
  if ((a = ArgPos(const_cast<char *>("-dim"), argc, argv)) > 0)
    n_hidden = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-seed"), argc, argv)) > 0)
    seed = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-verbose"), argc, argv)) > 0)
    verbosity = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-threads"), argc, argv)) > 0)
    n_threads = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-lr"), argc, argv)) > 0)
    initial_lr = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-nwalks"), argc, argv)) > 0)
    dw_n_walks = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-walklen"), argc, argv)) > 0)
    dw_walk_length = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-window"), argc, argv)) > 0)
    dw_window_size = atoi(argv[a + 1]);
#if INIT_HSM == HSM_INIT_PR
  if ((a = ArgPos(const_cast<char *>("-nprwalks"), argc, argv)) > 0)
    dw_window_size = atoi(argv[a + 1]);
#endif
  init_rng(seed);
  ifstream embFile(network_file, ios::in | ios::binary);
  if (embFile.is_open()) {
    char header[] = "----";
    embFile.seekg(0, ios::beg);
    embFile.read(header, 4);
    if (strcmp(header, "XGFS") != 0) {
      if (verbosity > 0)
        cout << "Invalid header!: " << header << endl;
      return 1;
    }
    embFile.read(reinterpret_cast<char *>(&nv), sizeof(long long));
    embFile.read(reinterpret_cast<char *>(&ne), sizeof(long long));
    offsets = static_cast<int *>(
        aligned_malloc((nv + 1) * sizeof(int32_t), DEFAULT_ALIGN));
    edges =
        static_cast<int *>(aligned_malloc(ne * sizeof(int32_t), DEFAULT_ALIGN));
    embFile.read(reinterpret_cast<char *>(offsets), nv * sizeof(int32_t));
    offsets[nv] = static_cast<int>(ne);
    embFile.read(reinterpret_cast<char *>(edges), sizeof(int32_t) * ne);
    if (verbosity > 0)
      cout << "nv: " << nv << ", ne: " << ne << endl;
    embFile.close();
  } else {
    return 0;
  }
  wVtx = static_cast<float *>(
      aligned_malloc(nv * n_hidden * sizeof(float), DEFAULT_ALIGN));
  for (int i = 0; i < nv * n_hidden; i++)
    wVtx[i] = (drand() - 0.5) / n_hidden;
  wCtx = static_cast<float *>(
      aligned_malloc(nv * n_hidden * sizeof(float), DEFAULT_ALIGN));
  memset(wCtx, 0, nv * n_hidden * sizeof(float));
  train_order =
      static_cast<int *>(aligned_malloc(nv * sizeof(int), DEFAULT_ALIGN));
  for (int i = 0; i < nv; i++)
    train_order[i] = i;
  shuffle(train_order, nv);
  hsm_weights =
      static_cast<float *>(aligned_malloc(nv * sizeof(float), DEFAULT_ALIGN));
#if INIT_HSM == HSM_INIT_DEGREE
  for (int i = 0; i < nv; i++)
    hsm_weights[i] = offsets[i + 1] - offsets[i];
#elif INIT_HSM == HSM_INIT_PR
  estimate_pr_rw(hsm_weights, nv * num_pr_walks);
#elif INIT_HSM == HSM_INIT_DW
  estimate_dw_probs(hsm_weights);
#else
#error Unknown INIT_HSM
#endif
  if (verbosity > 0)
#if VECTORIZE
    cout << "Using vectorized operations" << endl;
#else
    cout << "Not using vectorized operations (!)" << endl;
#endif
  init_hsm(hsm_weights);
  auto begin = chrono::steady_clock::now();
  Train();
  auto end = chrono::steady_clock::now();
  if (verbosity > 0)
    cout << endl
         << "Calculations took "
         << chrono::duration_cast<chrono::duration<float>>(end - begin).count()
         << " s to run" << endl;
  ofstream output(embedding_file, ios::binary);
  output.write(reinterpret_cast<char *>(wVtx), sizeof(float) * n_hidden * nv);
  output.close();
}