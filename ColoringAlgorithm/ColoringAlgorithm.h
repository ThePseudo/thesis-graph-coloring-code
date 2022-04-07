#ifndef _COLORING_ALGORITHM_H
#define _COLORING_ALGORITHM_H

#include "configuration.h"
#include "GraphRepresentation.h"
#include "AdjacencyMatrix.h"
#include "CompressedSparseRow.h"
#include <thread>

class ColoringAlgorithm {
protected:
	constexpr static int INVALID_COLOR = -1;

	GRAPH_REPR_T* _adj;
	std::vector<int> col;

public:
	static void printColorAlgorithmConfs();

	const GraphRepresentation& adj() const;
	const std::vector<int> getColors() const;

	virtual const int startColoring() = 0;

	// Receive vertex index, number of colors used so far and pointer where to save the color.
	// Returns the number of colors used
	const int computeVertexColor(size_t const v, int const n_cols, int* targetCol) const;

	std::vector<std::pair<size_t, size_t>> checkCorrectColoring();

	void printColors(std::ostream& os) const;
	void printDotFile(std::ostream& os) const;
};

#endif // !_COLORING_ALGORITHM_H