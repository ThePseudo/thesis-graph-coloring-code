#include "configuration.h"

#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
	std::vector<std::vector<int>> adj;
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

	const char* start;
	char* end;

	std::ifstream file;
	file.open(path);
	std::string line;
	if (file.is_open()) {
		for (int line_n = 0; file.good(); ++line_n) {
			std::getline(file, line);
			start = line.c_str();

			if (line_n == 0) {
				size_t cap = strtol(start, &end, 10);
				G.adj.resize(cap);
				G.col.resize(cap);
				G.recolor.resize(cap);

				std::fill(G.adj.begin(), G.adj.end(), std::vector<int>());
				std::fill(G.col.begin(), G.col.end(), INVALID_COLOR);
			} else {
				while (true) {
					size_t v = static_cast<int>(strtol(start, &end, 10));

					if (v == 0 && start == end) break;

					G.recolor[v] = v;
					++G.nV;

					start = end + 1;
					while (start != end) {
						size_t w = static_cast<int>(strtol(start, &end, 10));

						if (w == 0 && start == end) break;

						G.adj[v].push_back(w);
						++G.nE;

						start = end + 1;
					}
					// Reached end of line
				}
			}
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
		auto neighIt = G.adj[v].begin();
		auto forbidden = std::vector<bool>(n_cols);
		std::fill(forbidden.begin(), forbidden.end(), false);
		while (neighIt != G.adj[v].end()) {
			int w = *neighIt;
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