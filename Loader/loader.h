#ifndef _LOADER_H
#define _LOADER_H

#include "configuration.h"
#include <fstream>

bool parseInput(struct graph&, const char*);
#ifdef PARALLEL_INPUT_LOAD
void parseInputParallel(struct graph&, std::ifstream&);
#endif
#ifdef PARTITIONED_INPUT_LOAD
void parseInputParallel(struct graph&, std::string& const, int const, size_t const);
#endif

#endif // !_LOADER_H