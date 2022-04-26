#ifndef _JONES_PLASSMANN_H
#define _JONES_PLASSMANN_H

#include <cstdlib>
#include <ctime>

#include "configuration.h"

#include <vector>
#include <set>

#ifdef PARALLEL_GRAPH_COLOR
#include <atomic>
#include <thread>
#include <mutex>

#include "Barrier.h"
#endif

#include "GraphRepresentation.h"
#include "ColoringAlgorithm.h"

class JonesPlassmann : public ColoringAlgorithm {
private:
	std::vector<int> vWeights;
	int nIterations;

#ifdef SEQUENTIAL_GRAPH_COLOR
	std::vector<int> nWaits;
#endif
#ifdef PARALLEL_GRAPH_COLOR
	std::mutex mutex;
	std::vector<std::atomic_int> nWaits;
	std::vector<std::pair<size_t, size_t>> firstAndLasts;
	std::vector<int> n_colors;
	Barrier* barrier;

	void partitionVerticesByEdges(int const nThreads);
#endif

	const int solve();
	void coloringHeuristic(size_t const first, size_t const last, int& n_cols);
	void calcWaitTime(size_t const first, size_t const last);
	void colorWhileWaiting(size_t const first, size_t const last, int& n_cols);

public:
#ifdef PARALLEL_GRAPH_COLOR
	int MAX_THREADS_SOLVE = std::thread::hardware_concurrency();
#endif

	JonesPlassmann(std::string const filepath);
	const int startColoring() override;

	const int getIterations() const;

	const int colorWithCuda();
};

#endif // !_JONES_PLASSMANN_H