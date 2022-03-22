#include "configuration.h"
#include "loader.h"
#include "graph.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int detectConflicts(struct graph&);
void detectConflictsParallel(struct graph&, const int);
void sortGraphVerts(struct graph&);
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

	int n_cols = 0;

#ifdef SEQUENTIAL_GRAPH_COLOR
	sortGraphVerts(G);
	n_cols = colorGraph(G, n_cols);
#endif
#ifdef PARALLEL_GRAPH_COLOR
	int n_iters = 0;
	int n_confs = 0;

#ifdef PARALLEL_RECOLOR
	int partial_confs;
	do {
		sortGraphVerts(G);
		n_cols = colorGraph(G, n_cols);

		++n_iters;

		partial_confs = detectConflicts(G);
		n_confs += partial_confs;
	} while (partial_confs > 0);
#endif
#ifdef SEQUENTIAL_RECOLOR
	sortGraphVerts(G);
	n_cols = colorGraph(G, n_cols);
	++n_iters;
	n_confs = detectConflicts(G);

	if (n_confs > 0) {
		sortGraphVerts(G);
		int index = 0;

#ifdef COMPUTE_ELAPSED_TIME
		sampleTime();
#endif

		n_cols = colorGraphParallel(G, n_cols, index);

#ifdef COMPUTE_ELAPSED_TIME
		sampleTime();
		colorTime += getElapsedTime();
#endif
		++n_iters;
	}
#endif
#endif

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

int detectConflicts(struct graph& G) {
#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

	G.recolor.erase(G.recolor.begin(), G.recolor.end());
	std::vector<std::thread> threadPool;
	for (int i = 0; i < MAX_THREADS; ++i) {
		threadPool.emplace_back([&G, i] { detectConflictsParallel(G, i); });
	}

	for (auto& t : threadPool) {
		t.join();
	}

	int recolorSize = G.recolor.size();

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
	conflictsTime += getElapsedTime();
#endif

	return recolorSize;
}

void detectConflictsParallel(struct graph& G, const int i) {
	for (int v = i; v < G.nV; v += MAX_THREADS) {
		if (G.col[v] == INVALID_COLOR) {
			G.mutex.lock();
			G.recolor.push_back(v);
			G.mutex.unlock();
			continue;
		}

		for (
			auto neighIt = G.adj[v].begin();
			neighIt != G.adj[v].end();
			++neighIt
			) {
			int w = *neighIt;
			//if (v < w) continue;

			if (G.col[v] == G.col[w]) {
				G.mutex.lock();
				G.recolor.push_back(v);
				G.mutex.unlock();
				break;
			}
		}
	}

	return;
}

void sortGraphVerts(struct graph& G) {
#ifdef SORT_LARGEST_DEGREE_FIRST
	auto sort_lambda = [&G](const int v, const int w) { return G.adj[v].size() > G.adj[w].size(); };
#endif
#ifdef SORT_SMALLEST_DEGREE_FIRST
	auto sort_lambda = [&G](const int v, const int w) { return G.adj[v].size() < G.adj[w].size(); };
#endif
#ifdef SORT_VERTEX_ORDER
	auto sort_lambda = [&G](const int v, const int w) { return v < w; };
#endif
#ifdef SORT_VERTEX_ORDER_REVERSED
	auto sort_lambda = [&G](const int v, const int w) { return v > w; };
#endif

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

	std::sort(G.recolor.begin(), G.recolor.end(), sort_lambda);

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
	sortTime += getElapsedTime();
#endif
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