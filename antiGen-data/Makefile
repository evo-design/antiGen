CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -fopenmp

SRC := src/json.cpp src/tree.cpp src/hartigan.cpp src/sequences.cpp src/placement.cpp src/examples.cpp src/main.cpp
HDR := src/json.hpp src/tree.hpp src/hartigan.hpp src/alphabet.hpp src/sequences.hpp src/placement.hpp src/examples.hpp

antigen: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

clean:
	rm -f antigen

.PHONY: clean
