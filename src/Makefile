CC=g++
CXXFLAGS = -std=c++11 -march=native

ifeq ($(COMP),)
	CC=g++
	CXXFLAGS += -fopenmp -Ofast -Wno-write-strings
endif

ifeq ($(COMP),g++)
	CC=g++
	CXXFLAGS += -fopenmp -Ofast -Wno-write-strings
endif

ifeq ($(COMP),icpc)
	CC=icpc
	CXXFLAGS += -qopenmp -O3 -no-prec-div -ansi-alias -ip -static-intel
endif

ifeq ($(COMP),clang++)
	CC=clang++
	CXXFLAGS += -fopenmp -O3
endif

all: deepwalk

deepwalk: deepwalk.cpp
	$(CC) $(CXXFLAGS) deepwalk.cpp -o deepwalk

clean:
	rm deepwalk