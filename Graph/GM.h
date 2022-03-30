#ifndef _GRAPH_H
#define _GRAPH_H

#include "configuration.h"

#include <vector>
#include <mutex>

#include "GraphRepresentation.h"

class GM {
private:
	constexpr static int INVALID_COLOR = -1;

	GraphRepresentation* _adj;
	std::vector<int> col;
	std::vector<int> recolor;
	std::mutex mutex;

	int colorGraph(int);
	void sortGraphVerts();
#ifdef PARALLEL_GRAPH_COLOR
	int colorGraphParallel(int, int&);
	int detectConflicts();
	void detectConflictsParallel(const int);
#endif

public:
	const int MAX_THREADS_SOLVE = std::thread::hardware_concurrency();

	GM(GraphRepresentation& adj);

	const GraphRepresentation& adj();

#ifdef PARALLEL_GRAPH_COLOR
	const int solve(int&, int&);
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
	const int solve();
#endif
};

#endif // !_GRAPH_H
