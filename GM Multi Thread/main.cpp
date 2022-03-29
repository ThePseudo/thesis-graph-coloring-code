#include "configuration.h"
#include "loader.h"
#include "GM.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void printColors(struct graph&);
void printDotFile(struct graph&);

int main(int argc, char** argv) {
	graph G = { std::vector<std::vector<int>>(), std::vector<int>(), std::vector<int>(), 0, 0 };

	if (argc <= 1) {
		std::cout << "Usage: " << argv[0] << " <graph_path>" << std::endl;
		return 0;
	}

	std::cout << "Loading graph from " << argv[1] << std::endl;
	if (!parseInput(G, argv[1])) {
		std::cout << "An error occurred while loading the file." << std::endl << "The program will stop." << std::endl;
		return 0;
	}
	std::cout << "Graph succesfully loaded from file." << std::endl;
	std::cout << "Size: V: " << G.nV << ", E: " << G.nE << std::endl;
#ifndef SEQUENTIAL_GRAPH_COLOR
	std::cout << "Performing computation using " << MAX_THREADS << " threads." << std::endl;
#endif

#ifdef PARALLEL_GRAPH_COLOR
	int n_iters = 0;
	int n_confs = 0;
	auto cols = solve(G, n_iters, n_confs);
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
	auto cols = solve(G);
#endif
	int n_cols = cols.size();

	//printColors(G);
	//printDotFile(G);

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

	return 0;
}

void printColors(struct graph& G) {
	auto p = G.col.begin();
	for (int v = 0; v < G.nV; ++v) {
		std::cout << v << ": " << G.col[v] << std::endl;
	}
}

void printDotFile(struct graph& G) {
	std::ofstream file;
	file.open("output.dot");

	file << "strict graph {" << std::endl;
	file << "\tnode [colorscheme=pastel19]" << std::endl;
	// Write vertexes
	for (int v = 0; v < G.adj.size(); ++v) {
		file << "\t" << v << "[style=filled, color=" << G.col[v] + 1 << "]" << std::endl;
	}
	// Write edges
	for (int v = 0; v < G.adj.size(); ++v) {
		auto adjIt = G.adj[v].begin();
		while (adjIt != G.adj[v].end()) {
			int w = *adjIt;
			if (v <= w) {
				file << "\t" << v << " -- " << w << std::endl;
			}
			++adjIt;
		}
	}
	file << "}" << std::endl;
}