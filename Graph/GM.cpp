#include "configuration.h"
#include "GM.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include <algorithm>

GM::GM(GraphRepresentation& gr) {
	this->_adj = &gr;
	this->col = std::vector<int>(this->adj().nV());
	this->recolor = std::vector<int>(this->adj().nV());
	for (int i = 0; i < this->adj().nV(); ++i) {
		this->col[i] = GM::INVALID_COLOR;
		this->recolor[i] = i;
	}
}

const GraphRepresentation& GM::adj() {
	return *this->_adj;
}

int GM::colorGraph(int n_cols) {
#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

#ifdef PARALLEL_GRAPH_COLOR
	std::vector<std::thread> threadPool;
	int parallelIdx = 0;
	for (int i = 0; i < this->MAX_THREADS_SOLVE; ++i) {
		threadPool.emplace_back([&, n_cols] { this->colorGraphParallel(n_cols, parallelIdx); });
	}

	for (auto& t : threadPool) {
		t.join();
	}

	n_cols = *std::max_element(this->col.begin(), this->col.end()) + 1;
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
	for (
		auto it = this->recolor.begin();
		it != this->recolor.end();
		++it
		) {
		int v = *it;
		auto neighIt = this->adj().beginNeighs(v);
		auto forbidden = std::vector<bool>(n_cols);
		std::fill(forbidden.begin(), forbidden.end(), false);
		while (neighIt != this->adj().endNeighs(v)) {
			int w = *neighIt;
			int c = this->col[w];

			if (c != GM::INVALID_COLOR) forbidden[c] = true;
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

		this->col[v] = targetCol;
	}
#endif

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
	colorTime += getElapsedTime();
#endif

	return n_cols;
}

#ifdef PARALLEL_GRAPH_COLOR
int GM::colorGraphParallel(int n_cols, int& i) {
	this->mutex.lock();
	while (i < this->recolor.size()) {
		int v = this->recolor[i];
		++i;
		this->mutex.unlock();

		auto neighIt = this->adj().beginNeighs(v);
		auto forbidden = std::vector<bool>(n_cols);
		std::fill(forbidden.begin(), forbidden.end(), false);
		while (neighIt != this->adj().endNeighs(v)) {
			int w = *neighIt;
			int c = this->col[w];

			if (c != GM::INVALID_COLOR) {
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

		this->col[v] = targetCol;

		this->mutex.lock();
	}
	this->mutex.unlock();

	return n_cols;
}

int GM::detectConflicts() {
#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

	this->recolor.erase(this->recolor.begin(), this->recolor.end());
	std::vector<std::thread> threadPool;
	for (int i = 0; i < this->MAX_THREADS_SOLVE; ++i) {
		threadPool.emplace_back([&, i] { this->detectConflictsParallel(i); });
	}

	for (auto& t : threadPool) {
		t.join();
	}

	int recolorSize = this->recolor.size();

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
	conflictsTime += getElapsedTime();
#endif

	return recolorSize;
}

void GM::detectConflictsParallel(const int i) {
	for (int v = i; v < this->adj().nV(); v += this->MAX_THREADS_SOLVE) {
		if (this->col[v] == GM::INVALID_COLOR) {
			this->mutex.lock();
			this->recolor.push_back(v);
			this->mutex.unlock();
			continue;
		}

		for (
			auto neighIt = this->adj().beginNeighs(v);
			neighIt != this->adj().endNeighs(v);
			++neighIt
			) {
			int w = *neighIt;
			if (v < w) continue;

			if (this->col[v] == this->col[w]) {
				this->mutex.lock();
				this->recolor.push_back(v);
				this->mutex.unlock();
				break;
			}
		}
	}

	return;
}
#endif

void GM::sortGraphVerts() {
#ifdef SORT_LARGEST_DEGREE_FIRST
	auto sort_lambda = [&](const int v, const int w) { return this->adj().countNeighs(v) > this->adj().countNeighs(w); };
#endif
#ifdef SORT_SMALLEST_DEGREE_FIRST
	auto sort_lambda = [&](const int v, const int w) { return this->adj().countNeighs(v) < this->adj().countNeighs(w); };
#endif
#ifdef SORT_VERTEX_ORDER
	auto sort_lambda = [&](const int v, const int w) { return v < w; };
#endif
#ifdef SORT_VERTEX_ORDER_REVERSED
	auto sort_lambda = [&](const int v, const int w) { return v > w; };
#endif

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

	std::sort(this->recolor.begin(), this->recolor.end(), sort_lambda);

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
	sortTime += getElapsedTime();
#endif
}

#ifdef PARALLEL_GRAPH_COLOR
const int GM::solve(int& n_iters, int& n_confs) {
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
const int GM::solve() {
#endif
	int n_cols = 0;

#ifdef SEQUENTIAL_GRAPH_COLOR
	this->sortGraphVerts();
	n_cols = this->colorGraph(n_cols);
#endif
#ifdef PARALLEL_GRAPH_COLOR
	n_iters = 0;
	n_confs = 0;

#ifdef PARALLEL_RECOLOR
	int partial_confs;
	do {
		this->sortGraphVerts();
		n_cols = this->colorGraph(n_cols);

		++n_iters;

		partial_confs = this->detectConflicts();
		n_confs += partial_confs;
	} while (partial_confs > 0);
#endif
#ifdef SEQUENTIAL_RECOLOR
	this->sortGraphVerts();
	n_cols = this->colorGraph(n_cols);
	++n_iters;
	n_confs = this->detectConflicts();

	if (n_confs > 0) {
		this->sortGraphVerts();
		int index = 0;

#ifdef COMPUTE_ELAPSED_TIME
		sampleTime();
#endif

		n_cols = this->colorGraphParallel(n_cols, index);

#ifdef COMPUTE_ELAPSED_TIME
		sampleTime();
		colorTime += getElapsedTime();
#endif
		++n_iters;
	}
#endif
#endif

	return n_cols;
}