#ifndef _JONES_PLASSMANN_H
#define _JONES_PLASSMANN_H

#include <cstdlib>
#include <ctime>

#include <vector>
#include <set>
#include <thread>
#include <mutex>

#include "configuration.h"
#include "GraphRepresentation.h"
#include "ColoringAlgorithm.h"

class JonesPlassmann : public ColoringAlgorithm {
private:
	std::vector<float> vWeights;
	int nIterations;
	std::mutex mutex;

	const int solve();
	void findIndependentSet(std::set<size_t>::iterator& first, std::set<size_t>::iterator last, std::set<size_t>& indSet);

public:
	const int MAX_THREADS_SOLVE = std::thread::hardware_concurrency();

	JonesPlassmann(std::string const filepath);
	const int startColoring() override;

	const int getIterations() const;
};

#endif // !_JONES_PLASSMANN_H