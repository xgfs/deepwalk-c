//  deepwalk_test.cc

#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include "deepwalk.h"

TEST(SanityCheck, Expectation) {
    EXPECT_TRUE(true);
}

TEST(HuffmanTree, Construct) {
    float weights[] = {7, 15, 5, 6, 6};
    int indices[] = {1, 0, 4, 3, 2};

    huffman_tree ht(weights, indices, sizeof(weights) / sizeof(float));

    //  Compare branch encoding.
    uint8_t binary[] = {0, 1, 0, 1, 0, 0, 1, 1, 0, 0};

    for (int i = 0; i < sizeof(binary) / sizeof(int); ++i) {
        EXPECT_EQ(binary[i], ht.binary[i]);
    }

    //  Compare nodes weights.
    float count[] = {15, 7, 6, 6, 5, 11, 13, 24, 39, 1e25};

    for (int i = 0; i < sizeof(count) / sizeof(float); ++i) {
        EXPECT_EQ(count[i], ht.count[i]);
    }

    //  Compare indices of parent nodes.
    int parent[] = {8, 6, 6, 5, 5, 7, 7, 8, 0};

    for (int i = 0; i < sizeof(parent) / sizeof(int); ++i) {
        EXPECT_EQ(parent[i], ht.parent[i]);
    }
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
