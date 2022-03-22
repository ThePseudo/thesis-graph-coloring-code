#define SORT_LARGEST_DEGREE_FIRST
//#define SORT_SMALLEST_DEGREE_FIRST
//#define SORT_VERTEX_ORDER
//#define SORT_VERTEX_ORDER_REVERSED

#ifdef SORT_LARGEST_DEGREE_FIRST
#undef SORT_SMALLEST_DEGREE_FIRST
#undef SORT_VERTEX_ORDER
#undef SORT_VERTEX_ORDER_REVERSED 
#endif

#ifdef SORT_SMALLEST_DEGREE_FIRST
#undef SORT_LARGEST_DEGREE_FIRST
#undef SORT_VERTEX_ORDER
#undef SORT_VERTEX_ORDER_REVERSED 
#endif

#ifdef SORT_VERTEX_ORDER
#undef SORT_LARGEST_DEGREE_FIRST
#undef SORT_SMALLEST_DEGREE_FIRST
#undef SORT_VERTEX_ORDER_REVERSED 
#endif

#ifdef SORT_VERTEX_ORDER_REVERSED 
#undef SORT_LARGEST_DEGREE_FIRST
#undef SORT_SMALLEST_DEGREE_FIRST
#undef SORT_VERTEX_ORDER
#endif

#define PARALLEL_RECOLOR
//#define SEQUENTIAL_RECOLOR

#ifdef PARALLEL_RECOLOR
#undef SEQUENTIAL_RECOLOR
#endif

#ifdef SEQUENTIAL_RECOLOR
#undef PARALLEL_RECOLOR
#endif

#define COMPUTE_ELAPSED_TIME

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
float conflictsTime = 0.0f;

void sampleTime() {
	prev = now;
	now = clock();
}

float getElapsedTime() {
	return (float)(now - prev) / CLOCKS_PER_SEC;
}

#endif

constexpr auto INVALID_COLOR = -1;

__device__ std::vector<std::vector<int>> dAdj;

struct graph {
	std::vector<std::vector<int>> adj;
	std::vector<int> col;
	//std::vector<int> recolor;
	size_t nV, nE;
};

bool parseInput(struct graph&, const char*);
int detectConflicts(struct graph&);
void sortGraphVerts(struct graph&);
int colorGraph(struct graph&, int);
void printColors(struct graph&);
void printDotFile(struct graph&);

int main(int argc, char** argv) {
	graph G = { std::vector<std::vector<int>>(), std::vector<int>(), /*std::vector<int>(),*/ 0, 0 };

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

	//printColors(G);
	//printDotFile(G);

	std::cout << "Solution converged to in " << n_iters << " iterations." << std::endl;
	std::cout << "Detected a total of " << n_confs << " conflicts." << std::endl;
	std::cout << "Used a total of " << n_cols << " colors." << std::endl;

#ifdef COMPUTE_ELAPSED_TIME
	std::cout << std::endl << std::endl;
	std::cout << "TIME USAGE" << std::endl;
	std::cout << "File load:\t\t" << loadTime << " s" << std::endl;
	std::cout << "Vertex sort:\t\t" << sortTime << " s" << std::endl;
	std::cout << "Vertex color:\t\t" << colorTime << " s" << std::endl;
	std::cout << "Conflict search:\t" << conflictsTime << " s" << std::endl;
	std::cout << "Total:\t\t" << loadTime + sortTime + colorTime + conflictsTime << " s" << std::endl;
#endif

	return 0;
}

bool parseInput(struct graph& G, const char* path) {
	const char* start;
	char* end;
	std::ifstream file;

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

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
				//G.recolor.resize(cap);

				std::fill(G.adj.begin(), G.adj.end(), std::vector<int>());
				std::fill(G.col.begin(), G.col.end(), INVALID_COLOR);
			} else {
				while (true) {
					size_t v = static_cast<int>(strtol(start, &end, 10));

					if (v == 0 && start == end) break;

					//G.recolor[v] = v;
					++G.nV;

					start = end + 1;
					while (start != end) {
						size_t w = static_cast<int>(strtol(start, &end, 10));

						if (w == 0 && start == end) break;

						G.adj[v].push_back(w);
						G.adj[w].push_back(v);

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