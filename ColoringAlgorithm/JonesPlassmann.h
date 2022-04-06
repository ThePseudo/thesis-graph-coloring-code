#ifndef _JONES_PLASSMANN_H
#define _JONES_PLASSMANN_H

#include <cstdlib>
#include <ctime>

#include <atomic>
#include <vector>
#include <set>
#include <thread>
#include <mutex>

#include "Barrier.h"

#include "configuration.h"
#include "GraphRepresentation.h"
#include "ColoringAlgorithm.h"

class JonesPlassmann : public ColoringAlgorithm {
private:
	std::vector<float> vWeights;
	int nIterations;
	std::mutex mutex;
	Barrier* barrier;

	const int solve();
	void findIndependentSet(std::set<size_t>::iterator& first, std::set<size_t>::iterator last, std::set<size_t>& indSet);

	std::vector<std::atomic_int> nWaits;
	std::vector<std::pair<size_t, size_t>> firstAndLasts;
	void partitionVerticesByEdges(int const nThreads);
	void coloringHeuristic(size_t const first, size_t const last, int& n_cols, bool& again);
	void calcWaitTime(size_t const first, size_t const last);
	void colorWhileWaiting(size_t const first, size_t const last, int& n_cols, bool& again);

public:
	const int MAX_THREADS_SOLVE = std::thread::hardware_concurrency();

	JonesPlassmann(std::string const filepath);
	const int startColoring() override;

	const int getIterations() const;
};

#endif // !_JONES_PLASSMANN_H