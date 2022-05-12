#ifndef _GEBREMEDHIN_MANNE_H
#define _GEBREMEDHIN_MANNE_H

#include "configuration.h"

#include <vector>
#include <thread>
#include <mutex>
#include "Barrier.h"

#include "GraphRepresentation.h"
#include "ColoringAlgorithm.h"

class GebremedhinManne : public ColoringAlgorithm {
private:
	std::vector<int> recolor;
	std::mutex mutex;
	Barrier* barrier;

	int nConflicts;
	int nIterations;

	std::vector<std::vector<int>> colorClasses;


	int colorGraph(int n_cols);

	int performRecoloring(int n_cols);

	void partitionBasedColoring(int n_cols, int const initial, int const displacement);
	void improvedPartitionBasedColoring(int n_cols, int const initial, int const displacement);

	const int solve();
public:
	int MAX_THREADS_SOLVE = std::thread::hardware_concurrency();

	GebremedhinManne(std::string const filepath);
	const int startColoring() override;

	const int getConflicts() const;
	const int getIterations() const;
};

#endif // !_GEBREMEDHIN_MANNE_H
