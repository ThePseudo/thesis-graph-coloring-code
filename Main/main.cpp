#include "configuration.h"

#include "GebremedhinManne.h"
#include "JonesPlassmann.h"

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

void printColors(GebremedhinManne&);
void printDotFile(GebremedhinManne&);

int main(int argc, char** argv) {
#ifdef GRAPH_REPRESENTATION_ADJ_MATRIX
	AdjacencyMatrix* const _adj = new AdjacencyMatrix();
#endif
#ifdef GRAPH_REPRESENTATION_CSR
	CompressedSparseRow* const _adj = new CompressedSparseRow();
#endif
	auto& adj = *_adj;

	JonesPlassmann* _G;

	if (argc <= 1) {
		std::cout << "Usage: " << argv[0] << " <graph_path>" << std::endl;
		return 0;
	}

#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(0);
	bm.clear(1);
	bm.clear(2);
#ifdef PARALLEL_GRAPH_COLOR
	bm.clear(3);
#endif
#endif

	std::cout << "Loading graph from " << argv[1] << std::endl;

	std::ifstream fileIS;
	try {
		fileIS.open(argv[1]);
		std::istream& is = fileIS;
		is >> adj;
		_G = new JonesPlassmann(adj);
	} catch (const std::exception& e) {
		(void) e;
		std::cout << "An error occurred while loading the file." << std::endl << "The program will stop." << std::endl;
		if (fileIS.is_open()) {
			fileIS.close();
		}
		return 0;
	}
	if (fileIS.is_open()) {
		fileIS.close();
	}

	JonesPlassmann& G = *_G;

	std::cout << "Graph succesfully loaded from file." << std::endl;
	std::cout << "Size: V: " << adj.nV() << ", E: " << adj.nE() << std::endl;
#ifndef SEQUENTIAL_GRAPH_COLOR
	std::cout << "Performing computation using " << G.MAX_THREADS_SOLVE << " threads." << std::endl;
#endif

#ifdef PARALLEL_GRAPH_COLOR
	int n_iters = 0;
	int n_confs = 0;
	int n_cols = G.solve(n_iters, n_confs);
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
	int n_cols = G.solve();
#endif
	//int n_cols = cols.size();

#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Solution converged to in " << n_iters << " iterations." << std::endl;
	std::cout << "Detected a total of " << n_confs << " conflicts." << std::endl;
#endif
	std::cout << "Used a total of " << n_cols << " colors." << std::endl;

#ifdef COMPUTE_ELAPSED_TIME
	std::cout << std::endl << std::endl;
	std::cout << "TIME USAGE" << std::endl;
	std::cout << "File load:\t\t" << bm.getTimeOfFlag(0) << " s" << std::endl;
	std::cout << "Vertex sort:\t\t" << bm.getTimeOfFlag(1) << " s" << std::endl;
	std::cout << "Vertex color:\t\t" << bm.getTimeOfFlag(2) << " s" << std::endl;
#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Conflict search:\t" << bm.getTimeOfFlag(3) << " s" << std::endl;
#endif
	std::cout << "Total:\t\t" << bm.getTotalTime() << " s" << std::endl;
#endif

	//printColors(G);
	//printDotFile(G);

	return 0;
}

void printColors(GebremedhinManne& G) {
	auto col = G.getColors();
	auto p = col.begin();
	for (int v = 0; v < G.adj().nV(); ++v) {
		std::cout << v << ": " << col[v] << std::endl;
	}
}

void printDotFile(GebremedhinManne& G) {
	std::ofstream file;
	file.open("output.dot");

	auto col = G.getColors();

	file << "strict graph {" << std::endl;
	file << "\tnode [colorscheme=pastel19]" << std::endl;
	// Write vertexes
	for (int v = 0; v < G.adj().nV(); ++v) {
		file << "\t" << v << "[style=filled, color=" << col[v] + 1 << "]" << std::endl;
	}
	// Write edges
	for (size_t v = 0; v < G.adj().nV(); ++v) {
		auto adjIt = G.adj().beginNeighs(v);
		while (adjIt != G.adj().endNeighs(v)) {
			size_t w = *adjIt;
			if (v <= w) {
				file << "\t" << v << " -- " << w << std::endl;
			}
			++adjIt;
		}
	}
	file << "}" << std::endl;
}