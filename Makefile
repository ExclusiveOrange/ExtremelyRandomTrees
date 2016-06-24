# Makefile - Extremely Random Trees - Atlee Brink

CXX ?= g++
CXXFLAGS = -Wall -Wfatal-errors -O3 -std=c++11
RM ?= rm -f

PROGS = etgrow etpredict
DEPS = ndectree.hpp nexamples.hpp nextratrees.hpp nrandom.hpp nutil.hpp

all: $(PROGS)

all: etgrow etpredict

$(PROGS) : % : %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -lm $< -o $@

clean:
	$(RM) $(PROGS)
