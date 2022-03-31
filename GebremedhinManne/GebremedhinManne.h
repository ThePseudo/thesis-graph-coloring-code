#ifndef _GEBREMEFHIN_MANNE_H
#define _GEBREMEFHIN_MANNE_H

#include "configuration.h"

#include <vector>
#include <thread>
#include <mutex>

#include "GraphRepresentation.h"

class GebremedhinManne {
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

	GebremedhinManne(GraphRepresentation& adj);

	const GraphRepresentation& adj();

#ifdef PARALLEL_GRAPH_COLOR
	const int solve(int&, int&);
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
	const int solve();
#endif

	const std::vector<int> getColors();
};

#endif // !_GEBREMEFHIN_MANNE_H
