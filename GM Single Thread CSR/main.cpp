#include "configuration.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "CompressedSparseRow.h"

#ifdef COMPUTE_ELAPSED_TIME
#include <ctime>

clock_t prev;
clock_t now;

float loadTime = 0.0f;
float sortTime = 0.0f;
float colorTime = 0.0f;

void sampleTime() {
	prev = now;
	now = clock();
}

float getElapsedTime() {
	return (float)(now - prev) / CLOCKS_PER_SEC;
}

#endif

constexpr auto INVALID_COLOR = -1;

struct graph {
	CompressedSparseRow adj;
	std::vector<int> col;
	std::vector<int> recolor;
	size_t nV, nE;
};

bool parseInput(struct graph&, const char*);
void sortGraphVerts(struct graph&);
int colorGraph(struct graph&, int);
void printColors(struct graph&);
void printDotFile(struct graph&);

int main(int argc, char** argv) {
	graph G = { CompressedSparseRow(), std::vector<int>(), std::vector<int>(), 0, 0 };

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

	sortGraphVerts(G);
	n_cols = colorGraph(G, n_cols);
	
	//printColors(G);
	//printDotFile(G);

	std::cout << "Used a total of " << n_cols << " colors." << std::endl;

#ifdef COMPUTE_ELAPSED_TIME
	std::cout << std::endl << std::endl;
	std::cout << "TIME USAGE" << std::endl;
	std::cout << "File load:\t\t" << loadTime << " s" << std::endl;
	std::cout << "Vertex sort:\t\t" << sortTime << " s" << std::endl;
	std::cout << "Vertex color:\t\t" << colorTime << " s" << std::endl;
	std::cout << "Total:\t\t" << loadTime + sortTime + colorTime << " s" << std::endl;
#endif

	return 0;
}


bool parseInput(struct graph& G, const char* path) {
#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

	std::ifstream file;
	file.open(path);
	if (file.good()) {
		file >> G.adj;
		G.col.resize(G.adj.countRows());
		std::fill(G.col.begin(), G.col.end(), INVALID_COLOR);
		G.recolor.resize(G.adj.countRows());
		G.nV = G.adj.countRows();
		G.nE = G.adj.count();

		for (int i = 0; i < G.nV; ++i) {
			G.recolor[i] = i;
		}
	} else {
		file.close();
		return false;
	}
	
	file.close();

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
	loadTime += getElapsedTime();
#endif

	return true;
}

void sortGraphVerts(struct graph& G) {
#ifdef SORT_LARGEST_DEGREE_FIRST
	auto sort_lambda = [&G](const int v, const int w) { return G.adj.endRow(v) - G.adj.beginRow(v) > G.adj.endRow(w) - G.adj.beginRow(w); };
#endif
#ifdef SORT_SMALLEST_DEGREE_FIRST
	auto sort_lambda = [&G](const int v, const int w) { return G.adj.endRow(v) - G.adj.beginRow(v) < G.adj.endRow(w) - G.adj.beginRow(w) };
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

int colorGraph(struct graph& G, int n_cols) {
#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

	for (
		auto it = G.recolor.begin();
		it != G.recolor.end();
		++it
		) {
		int v = *it;
		auto neighIt = G.adj.beginRow(v);
		auto forbidden = std::vector<bool>(n_cols);
		std::fill(forbidden.begin(), forbidden.end(), false);
		while (neighIt != G.adj.endRow(v)) {
			int const w = *neighIt;
			int c = G.col[w];

			if (c != INVALID_COLOR) forbidden[c] = true;
			++neighIt;
		}
		auto targetIt = std::find(forbidden.begin(), forbidden.end(), false);
		int targetCol;
		if (targetIt == forbidden.end()) {
			// All forbidden. Add new color
			targetCol = n_cols++;
		} else {
			targetCol = targetIt - forbidden.begin();
		}

		G.col[v] = targetCol;
	}

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
	colorTime += getElapsedTime();
#endif

	return n_cols;
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
	for (int v = 0; v < G.nV; ++v) {
		file << "\t" << v << "[style=filled, color=" << G.col[v] + 1 << "]" << std::endl;
	}
	// Write edges
	for (int v = 0; v < G.nV; ++v) {
		auto adjIt = G.adj.beginRow(v);
		while (adjIt != G.adj.endRow(v)) {
			int w = *adjIt;
			if (v <= w) {
				file << "\t" << v << " -- " << w << std::endl;
			}
			++adjIt;
		}
	}
	file << "}" << std::endl;
}