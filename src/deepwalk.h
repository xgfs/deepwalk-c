//  deepwalk.h
//  
//  Describe public interface and types.

#pragma once

#include <algorithm>

//  Intermidiate entry point in order to test computational routines.
int run(int argc, char **argv);

//  Represent Huffman tree and provide routines to build from weights. It
//  allows test critical section independently. Also it is a bit closer to
//  RAII since free memory on object destroying.
struct huffman_tree {
    int8_t *    binary;
    float *     count;
    int *       parent;
    size_t      noleaves;

    huffman_tree(const float *weights, const int *indices, size_t nv)
        : binary(nullptr)
        , count(nullptr)
        , parent(nullptr)
        , noleaves(nv)
    {
        size_t length = 2 * nv + 1;

        binary = new int8_t[length];
        count = new float[length];
        parent = new int[length];

        //  TODO: actually it is not necessary
        std::fill(binary, binary + length, 0);
        std::fill(count, count + length, 0);
        std::fill(parent, parent + length, 0);

        make(weights, indices, nv);
    }

    ~huffman_tree(void) {
        delete[] binary;
        delete[] count;
        delete[] parent;
    }

    void make(const float *weights, const int *indices, size_t nv);
};
