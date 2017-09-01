TARGET = deepwalk-test

SOURCES = ../src/deepwalk.cpp
OBJECTS = $(SOURCES:.cpp=.o)

CXXFLAGS = -std=c++11 -march=native -g -I../src
CXXFLAGS += $(shell gtest-config --cppflags --cxxflags)
LDFLAGS += $(shell gtest-config --ldflags --libs)

ifeq ($(CXX), g++)
	CXXFLAGS += -Ofast
else ifeq ($(CXX), icpc)
	CXXFLAGS += -qopenmp -O3 -no-prec-div -ansi-alias -ip -static-intel
else ifeq ($(CXX), clang++)
	CXXFLAGS += -fopenmp=libomp -O3 -Wno-shift-op-parentheses
else
	# use g++ by default
	CXX = g++
	CXXFLAGS += -fopenmp -Ofast
endif


all: $(TARGET)

$(TARGET): $(OBJECTS) main.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -rfv $(OBJECTS) $(TARGET)

.PHONY: all clean test
