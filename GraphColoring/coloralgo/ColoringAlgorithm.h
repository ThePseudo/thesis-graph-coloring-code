#ifndef _COLORING_ALGORITHM_H
#define _COLORING_ALGORITHM_H

//#include "configuration.h"
#include "GraphRepresentation.h"
#ifdef GRAPH_REPRESENTATION_ADJ_MATRIX
#include "AdjacencyMatrix.h"
#endif
#ifdef GRAPH_REPRESENTATION_CSR
#include "CompressedSparseRow.h"
#endif
#include <thread>

#if !defined(_MSC_VER) || _MSC_VER < 1400
#define define_super(c) typedef c __super
#else
#define define_super(c)
#endif

class ColoringAlgorithm {
protected:
        static int INVALID_COLOR;

	int resetCount;
	GRAPH_REPR_T* _adj = nullptr;
	std::vector<int> col;

public:
	static void printColorAlgorithmConfs();

	const GRAPH_REPR_T& adj() const;
	const std::vector<int> getColors() const;

	virtual void init();
	virtual void reset();
	virtual const int startColoring() = 0;

	// Receive vertex index, number of colors used so far and pointer where to save the color.
	// Returns the number of colors used
	const int computeVertexColor(int const v, int const n_cols, int* targetCol) const;

	std::vector<std::pair<int, int>> checkCorrectColoring();

	virtual void printExecutionInfo() const = 0;
	virtual void printBenchmarkInfo() const;

	void printColors(std::ostream& os) const;
	void printDotFile(std::ostream& os) const;
};

#endif // !_COLORING_ALGORITHM_H
