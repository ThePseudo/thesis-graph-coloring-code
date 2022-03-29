#include "configuration.h"
#include "GM.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include <algorithm>

int colorGraph(struct graph&, int);
void sortGraphVerts(struct graph&);
#ifdef PARALLEL_GRAPH_COLOR
int colorGraphParallel(struct graph&, int, int&);
int detectConflicts(struct graph&);
void detectConflictsParallel(struct graph&, const int);
#endif

int colorGraph(struct graph& G, int n_cols) {
#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

#ifdef PARALLEL_GRAPH_COLOR
	std::vector<std::thread> threadPool;
	int parallelIdx = 0;
	for (int i = 0; i < MAX_THREADS; ++i) {
		threadPool.emplace_back([&G, n_cols, &parallelIdx] { colorGraphParallel(G, n_cols, parallelIdx); });
	}

	for (auto& t : threadPool) {
		t.join();
	}

	n_cols = *std::max_element(G.col.begin(), G.col.end()) + 1;
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
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
#endif

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
	colorTime += getElapsedTime();
#endif

	return n_cols;
}

#ifdef PARALLEL_GRAPH_COLOR
int colorGraphParallel(struct graph& G, int n_cols, int& i) {
	G.mutex.lock();
	while (i < G.recolor.size()) {
		int v = G.recolor[i];
		++i;
		G.mutex.unlock();

		auto neighIt = G.adj[v].begin();
		auto forbidden = std::vector<bool>(n_cols);
		std::fill(forbidden.begin(), forbidden.end(), false);
		while (neighIt != G.adj[v].end()) {
			int w = *neighIt;
			int c = G.col[w];

			if (c != INVALID_COLOR) {
				if (c >= n_cols) {
					n_cols = c + 1;
					forbidden.resize(n_cols, false);
				}
				forbidden[c] = true;
			}
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

		G.mutex.lock();
	}
	G.mutex.unlock();

	return n_cols;
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
#endif

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

#ifdef PARALLEL_GRAPH_COLOR
std::vector<int> solve(struct graph& G, int& n_iters, int& n_confs) {
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
std::vector<int> solve(struct graph& G) {
#endif
	int n_cols = 0;

#ifdef SEQUENTIAL_GRAPH_COLOR
	sortGraphVerts(G);
	n_cols = colorGraph(G, n_cols);
#endif
#ifdef PARALLEL_GRAPH_COLOR
	n_iters = 0;
	n_confs = 0;

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

	return G.col;
}