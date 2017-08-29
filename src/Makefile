TARGET = deepwalk

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

CXXFLAGS = -std=c++11 -march=native

ifeq ($(CXX), g++)
	CXXFLAGS += -fopenmp -Ofast -Wno-write-strings
else ifeq ($(CXX), icpc)
	CXXFLAGS += -qopenmp -O3 -no-prec-div -ansi-alias -ip -static-intel
else ifeq ($(CXX), clang++)
	CXXFLAGS += -fopenmp -O3
else
	# use g++ by default
	CXX = g++
	CXXFLAGS += -fopenmp -Ofast -Wno-write-strings
endif

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -rfv $(OBJECTS) $(TARGET)

.PHONY: all clean
