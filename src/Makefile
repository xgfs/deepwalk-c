CC=g++
CXXFLAGS = -std=c++11 -march=native

ifeq ($(COMP),)
	CC=g++
endif

ifeq ($(COMP),g++)
	CC=g++
	CXXFLAGS += -fopenmp -Ofast -mavx
endif

ifeq ($(COMP),icpc)
	CC=icpc
	CXXFLAGS += -qopenmp -O3 -no-prec-div -ansi-alias -axavx -ansi-alias -ip -march=native
endif

ifeq ($(COMP),clang++)
	CC=clang++
	CXXFLAGS += -fopenmp -O3 -mavx -march=native
endif

all: hello

deepwalk: deepwalk.cpp
    $(CC) $(CFLAGS) deepwalk.cpp -o deepwalk

clean:
    rm *o deepwalk