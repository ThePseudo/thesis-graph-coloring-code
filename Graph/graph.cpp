#include "configuration.h"
#include "graph.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include <algorithm>

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
#endif