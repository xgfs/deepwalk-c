TARGET = deepwalk

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

CXXFLAGS = -std=c++11 -march=native

OS := $(shell uname)

ifeq ($(OS),Darwin)
	CXXFLAGS += -Xpreprocessor
	LIB = -lomp
else
	LIB =
endif

ifeq ($(CXX), g++)
	CXXFLAGS += -fopenmp -Ofast
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

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIB)

clean:
	rm -rfv $(OBJECTS) $(TARGET)

.PHONY: all clean
