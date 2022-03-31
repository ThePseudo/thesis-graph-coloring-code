#ifndef _JONES_PLASSMANN_H
#define _JONES_PLASSMANN_H

#include <cstdlib>
#include <ctime>

#include <vector>
#include <thread>

#include "configuration.h"
#include "GraphRepresentation.h"

class JonesPlassmann {
private:
	constexpr static int INVALID_COLOR = -1;

	GraphRepresentation* _adj;
	std::vector<int> col;
	std::vector<float> vWeights;

public:
	const int MAX_THREADS_SOLVE = std::thread::hardware_concurrency();

	JonesPlassmann(GraphRepresentation& adj);

	const GraphRepresentation& adj();

	const int solve();
	const std::vector<int> getColors();
};

#endif // !_JONES_PLASSMANN_H