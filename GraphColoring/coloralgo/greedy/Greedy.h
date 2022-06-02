#ifndef _GREEDY_H
#define _GREEDY_H

#ifdef COLORING_ALGORITHM_GREEDY

#include <vector>
#include <thread>
#include <mutex>

#include "GraphRepresentation.h"
#include "ColoringAlgorithm.h"

class Greedy : public ColoringAlgorithm {

	define_super(ColoringAlgorithm);

private:
	std::vector<int> recolor;
	std::mutex mutex;

	int nConflicts;
	int nIterations;


	int colorGraph(int);
	void sortGraphVerts();

	int colorGraphParallel(int, int&);
	int detectConflicts();
	void detectConflictsParallel(const int);

	const int solve();
public:
	int MAX_THREADS_SOLVE = std::thread::hardware_concurrency();

	Greedy(std::string const filepath);

	void init() override;
	void reset() override;
	const int startColoring() override;
	
	void printExecutionInfo() const override;
	void printBenchmarkInfo() const override;

	const int getConflicts() const;
	const int getIterations() const;
};

#endif

#endif // !_GREEDY_H
