#ifndef _LOADER_H
#define _LOADER_H

#include "configuration.h"
#include <fstream>

bool parseInput(struct graph&, const char*);
#ifdef PARALLEL_INPUT_LOAD
void parseInputParallel(struct graph&, std::ifstream&);
#endif

#endif // !_LOADER_H