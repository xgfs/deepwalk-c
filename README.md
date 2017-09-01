# deepwalk-c

DeepWalk implementation in C++. DeepWalk uses short random walks to learn representations for vertices in unweighted graphs.

## Installation and usage

For C++ executable:

    cd src && make;

should be enough on most platforms. If you need to change the default compiler (i.e. to Intel), use:

    make CXX=icpc

Intel® FMA availability is crucial for performance of the implementation, meaning the processor  Haswell (2013). You will get a warning on runtime if your processor does not support it.

### Usage

```
Usage: deepwalk [OPTIONS]

Options:
  -input PATH                    Input file in binary CSR format
  -output PATH                   Output file, written in binary
  -threads INT                   Number of threads to use (default 1)
                                   Note: hyperthreading helps as well
  -dim INT                       DeepWalk parameter d: dimensionality of
                                   embeddings (default 128)
  -nwalks INT                    DeepWalk parameter gamma: number of walks per
                                   node (default 80)
  -walklen INT                   DeepWalk parameter t: length of random walk
                                   from each node(default 80)
  -window INT                    DeepWalk parameter w: window size (default 10)
  -nprwalks INT                  Implementation parameter w: number of random
                                   walks for HSM tree (default 100)
  -lr FLOAT                      Initial learning rate
  -seed INT                      Sets the random number generator seed to INT
  -verbose INT                   Controls verbosity level in [0,1,2], 0 meaning
                                   nothing will be displayed, and 2 mening
                                   training progress will be displayed.
```

### Graph format

This implementation uses a custom graph format, namely binary [compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) (BCSR) format for efficiency and reduced memory usage. Converter for three common graph formats (MATLAB sparse matrix, adjacency list, edge list) can be found in the root directory of the project. Usage:

```
$ convert-bcsr --help
Usage: convert-bcsr [OPTIONS] INPUT OUTPUT

  Converter for three common graph formats (MATLAB sparse matrix, adjacency
  list, edge list) can be found in the root directory of the project.

Options:
  --format [mat|edgelist|adjlist]
                                  File format of input file
  --matfile-variable-name TEXT    variable name of adjacency matrix inside a
                                  .mat file.
  --undirected / --directed       Treat graph as undirected.
  --sep TEXT                      Separator of input file
  --help                          Show this message and exit.
```

1. ``--format adjlist`` for an adjacency list, e.g:

        1 2 3 4 5 6 7 8 9 11 12 13 14 18 20 22 32
        2 1 3 4 8 14 18 20 22 31
        3 1 2 4 8 9 10 14 28 29 33
        ...

1. ``--format edgelist`` for an edge list, e.g:

        1 2
        1 3
        1 4
        ...

1. ``--format mat`` for a Matlab MAT file containing an adjacency matrix
        (note, you must also specify the variable name of the adjacency matrix ``--matfile-variable-name``)

## Why

[Official implementation](https://github.com/phanein/deepwalk) of DeepWalk is not maintained. To reproduce the results, one would need to install very old scipy and gensim versions. That confuses researchers and other people wanting to tinker with the code. Also, Cython word2vec implementation is [known to scale worse with the number of cores](https://github.com/RaRe-Technologies/gensim/issues/1291).

## Implementation differences

Even though the algorithm claims to be 'online', hierarchical softmax tree must be constructed from the random walk corpora prior to training. Here, we run a serie of random walks and count vertex statistics to initialize the hierarchical softmax tree. The original strategy can be used by compiling with the INIT_HSM=2 flag.

## Performance

Measured on i7-5930k. Blogcatalog graph (n=10312) with default parameters (gamma=80, t=80, d=128, w=10):

| Version | 6 threads  | 12 threads |
| --- | --- | --- |
| Original (cython) | 395.41s  | 480.13s |
| This (c++) | 137.34s | 112.32s |

Keep in mind that the original DeepWalk implementation keeps all the walks either in memory or on disk! Another benchmark with parameter set from node2vec paper (gamma=10, t=40, d=128, w=10) (meaning >16x less training data):

| Version | 6 threads  | 12 threads |
| --- | --- | --- |
| Original (cython) | 27.28s | 32.24s |
| This (c++) | 8.34s | 6.84s |

## Citing

If you find DeepWalk useful in your research, we ask that you cite the original paper:

    @inproceedings{Perozzi:2014:DOL:2623330.2623732,
        author = {Perozzi, Bryan and Al-Rfou, Rami and Skiena, Steven},
        title = {DeepWalk: Online Learning of Social Representations},
        booktitle = {Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
        series = {KDD '14},
        year = {2014},
        isbn = {978-1-4503-2956-9},
        location = {New York, New York, USA},
        pages = {701--710},
        numpages = {10},
        url = {http://doi.acm.org/10.1145/2623330.2623732},
        doi = {10.1145/2623330.2623732},
        acmid = {2623732},
        publisher = {ACM},
        address = {New York, NY, USA},
        keywords = {deep learning, latent representations, learning with partial labels, network classification, online learning, social networks},
    }

## Contact

`echo "%7=87.=<2=<>527@192.()" | tr '#-)/->' '_-|'`
