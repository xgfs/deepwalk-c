//  deepwalk_test.cc

#include <memory>
#include <gtest/gtest.h>
#include "deepwalk.h"

TEST(SanityCheck, Expectation) {
    EXPECT_TRUE(true);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
