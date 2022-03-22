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

#endif // !_GRAPH_H
