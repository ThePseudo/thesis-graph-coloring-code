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
	
#ifdef PARALLEL_GRAPH_COLOR
	std::mutex mutex;
	std::vector<std::atomic_int> nWaits;
	std::vector<std::pair<size_t, size_t>> firstAndLasts;
	//std::vector<int> n_colors;
	Barrier* barrier;

	void partitionVerticesByEdges(int const nThreads);
	void coloringHeuristic(size_t const first, size_t const last, int& n_cols);
	void calcWaitTime(size_t const first, size_t const last);
	void colorWhileWaiting(size_t const first, size_t const last, int& n_cols);
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
	void findIndependentSet(std::set<size_t>::iterator& first, std::set<size_t>::iterator last, std::set<size_t>& indSet);
#endif

	const int solve();

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