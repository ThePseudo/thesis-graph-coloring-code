#ifndef _GREEDY_H
#define _GREEDY_H

#include "configuration.h"

#include <vector>
#include <thread>
#include <mutex>

#include "GraphRepresentation.h"
#include "ColoringAlgorithm.h"

class Greedy : public ColoringAlgorithm {
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
	const int startColoring() override;

	const int getConflicts() const;
	const int getIterations() const;
};

#endif // !_GREEDY_H

