#ifndef _GEBREMEFHIN_MANNE_H
#define _GEBREMEFHIN_MANNE_H

#include "configuration.h"

#include <vector>
#include <thread>
#include <mutex>

#include "GraphRepresentation.h"
#include "ColoringAlgorithm.h"

class GebremedhinManne : public ColoringAlgorithm {
private:
	std::vector<int> recolor;
	std::mutex mutex;
#ifdef PARALLEL_GRAPH_COLOR
	int nConflicts;
	int nIterations;
#endif

	int colorGraph(int);
	void sortGraphVerts();
#ifdef PARALLEL_GRAPH_COLOR
	int colorGraphParallel(int, int&);
	int detectConflicts();
	void detectConflictsParallel(const int);
#endif

//#ifdef PARALLEL_GRAPH_COLOR
//	const int solve(int&, int&);
//#endif
//#ifdef SEQUENTIAL_GRAPH_COLOR
	const int solve();
//#endif
public:
	const int MAX_THREADS_SOLVE = std::thread::hardware_concurrency();

	GebremedhinManne(std::string const filepath);
	const int startColoring() override;

#ifdef PARALLEL_GRAPH_COLOR
	const int getConflicts() const;
	const int getIterations() const;
#endif
};

#endif // !_GEBREMEFHIN_MANNE_H
