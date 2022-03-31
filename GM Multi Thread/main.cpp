#include "configuration.h"
#include "GM.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include "GraphRepresentation.h"
#ifdef GRAPH_REPRESENTATION_ADJ_MATRIX
#include "AdjacencyMatrix.h";
#endif
#ifdef GRAPH_REPRESENTATION_CSR
#include "CompressedSparseRow.h";
#endif

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void printColors(GM&);
void printDotFile(GM&);

int main(int argc, char** argv) {
#ifdef GRAPH_REPRESENTATION_ADJ_MATRIX
	AdjacencyMatrix* const _adj = new AdjacencyMatrix();
#endif
#ifdef GRAPH_REPRESENTATION_CSR
	CompressedSparseRow* const _adj = new CompressedSparseRow();
#endif
	auto& adj = *_adj;

	GM* _G;

	if (argc <= 1) {
		std::cout << "Usage: " << argv[0] << " <graph_path>" << std::endl;
		return 0;
	}

	std::cout << "Loading graph from " << argv[1] << std::endl;

	std::ifstream fileIS;
	try {
		fileIS.open(argv[1]);
		std::istream& is = fileIS;
		is >> adj;
		_G = new GM(adj);
	} catch (const std::exception& e) {
		std::cout << "An error occurred while loading the file." << std::endl << "The program will stop." << std::endl;
		if (fileIS.is_open()) {
			fileIS.close();
		}
		return 0;
	}
	if (fileIS.is_open()) {
		fileIS.close();
	}

	GM& G = *_G;

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
	std::cout << "File load:\t\t" << loadTime << " s" << std::endl;
	std::cout << "Vertex sort:\t\t" << sortTime << " s" << std::endl;
	std::cout << "Vertex color:\t\t" << colorTime << " s" << std::endl;
#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Conflict search:\t" << conflictsTime << " s" << std::endl;
#endif
	std::cout << "Total:\t\t" << loadTime + sortTime + colorTime + conflictsTime << " s" << std::endl;
#endif

	//printColors(G);
	//printDotFile(G);

	return 0;
}

void printColors(GM& G) {
	auto col = G.getColors();
	auto p = col.begin();
	for (int v = 0; v < G.adj().nV(); ++v) {
		std::cout << v << ": " << col[v] << std::endl;
	}
}

void printDotFile(GM& G) {
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
	for (int v = 0; v < G.adj().nV(); ++v) {
		auto adjIt = G.adj().beginNeighs(v);
		while (adjIt != G.adj().endNeighs(v)) {
			int w = *adjIt;
			if (v <= w) {
				file << "\t" << v << " -- " << w << std::endl;
			}
			++adjIt;
		}
	}
	file << "}" << std::endl;
}