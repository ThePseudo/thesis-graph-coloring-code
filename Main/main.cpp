#include "configuration.h"

#include "ColoringAlgorithm.h"
#ifdef COLORING_ALGORITHM_GM
#include "GebremedhinManne.h"
#endif
#ifdef COLORING_ALGORITHM_JP
#include "JonesPlassmann.h"
#endif

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include "GraphRepresentation.h"
#ifdef GRAPH_REPRESENTATION_ADJ_MATRIX
#include "AdjacencyMatrix.h"
#endif
#ifdef GRAPH_REPRESENTATION_CSR
#include "CompressedSparseRow.h"
#endif

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void printColors(ColoringAlgorithm&);
void printDotFile(ColoringAlgorithm&);

int main(int argc, char** argv) {
	COLORING_ALGO_T* _G;
	if (argc <= 1) {
		std::cout << "Usage: " << argv[0] << " <graph_path>" << std::endl;
		return 0;
	}

	std::cout << "Loading graph from " << argv[1] << std::endl;

	_G = new COLORING_ALGO_T(std::string(argv[1]));

	COLORING_ALGO_T& G = *_G;
	
	std::cout << "Graph succesfully loaded from file." << std::endl;
	std::cout << "Size: V: " << G.adj().nV() << ", E: " << G.adj().nE() << std::endl;
#ifndef SEQUENTIAL_GRAPH_COLOR
#ifdef COLORING_ALGORITHM_GM
	std::cout << "Performing computation using " << G.MAX_THREADS_SOLVE << " threads." << std::endl;
#endif
#endif

	int n_cols = G.startColoring();

#if defined(COLORING_ALGORITHM_GM) && defined(PARALLEL_GRAPH_COLOR)
	std::cout << "Solution converged to in " << G.getIterations() << " iterations." << std::endl;
	std::cout << "Detected a total of " << G.getConflicts() << " conflicts." << std::endl;
#endif
#ifdef COLORING_ALGORITHM_JP
	std::cout << "Solution converged to in " << G.getIterations() << " iterations." << std::endl;
#endif
	std::cout << "Used a total of " << n_cols << " colors." << std::endl;

#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	std::cout << std::endl << std::endl;
	std::cout << "TIME USAGE" << std::endl;
	std::cout << "File load:\t\t" << bm.getTimeOfFlag(0) << " s" << std::endl;
#ifdef COLORING_ALGORITHM_JP
	std::cout << "Ind set create:\t\t" << bm.getTimeOfFlag(1) << " s" << std::endl;
	std::cout << "Vertex color:\t\t" << bm.getTimeOfFlag(2) << " s" << std::endl;
#endif
#ifdef COLORING_ALGORITHM_GM
	std::cout << "Vertex sort:\t\t" << bm.getTimeOfFlag(1) << " s" << std::endl;
	std::cout << "Vertex color:\t\t" << bm.getTimeOfFlag(2) << " s" << std::endl;
#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Conflict search:\t" << bm.getTimeOfFlag(3) << " s" << std::endl;
#endif
#endif
	std::cout << "Total:\t\t" << bm.getTotalTime() << " s" << std::endl;
#endif

	//G.printColors(std::out);
	//G.printDotFile(std::ofstream("output.txt"));

	return 0;
}