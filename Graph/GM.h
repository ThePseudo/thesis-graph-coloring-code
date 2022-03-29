#ifndef _GRAPH_H
#define _GRAPH_H

#include "configuration.h"

#include <vector>
#include <mutex>

const auto MAX_THREADS = std::thread::hardware_concurrency();

constexpr auto INVALID_COLOR = -1;

struct graph {
	std::vector<std::vector<int>> adj;
	std::vector<int> col;
	std::vector<int> recolor;
	size_t nV, nE;
	std::mutex mutex;
};

#ifdef PARALLEL_GRAPH_COLOR
std::vector<int> solve(struct graph&, int&, int&);
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
std::vector<int> solve(struct graph&);
#endif

#endif // !_GRAPH_H
