#include "configuration.h"
#include "GebremedhinManne.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include <algorithm>
#include <fstream>

GebremedhinManne::GebremedhinManne(std::string const filepath) {
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(0);

	this->_adj = new GRAPH_REPR_T();

	std::ifstream fileIS;
	fileIS.open(filepath);
	std::istream& is = fileIS;
	is >> *this->_adj;

	if (fileIS.is_open()) {
		fileIS.close();
	}

	this->MAX_THREADS_SOLVE = std::min(this->adj().nV(), this->MAX_THREADS_SOLVE);
	this->col = std::vector<int>(this->adj().nV(), GebremedhinManne::INVALID_COLOR);
	int const maxConflicts = this->adj().nE() / this->adj().nV() * (this->MAX_THREADS_SOLVE - 1) / 2;
	this->recolor = std::vector<int>(maxConflicts);

	this->nConflicts = 0;
	this->nIterations = 0;

	
	this->barrier = new Barrier(this->MAX_THREADS_SOLVE);
}

const int GebremedhinManne::startColoring() {
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(1);
	bm.clear(2);
#ifdef PARALLEL_GRAPH_COLOR
	bm.clear(3);
#endif

	return this->solve();
}

int GebremedhinManne::colorGraph(int n_cols) {
#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();
#endif

#ifdef PARALLEL_GRAPH_COLOR
	std::vector<std::thread> threadPool;
	for (int i = 0; i < this->MAX_THREADS_SOLVE; ++i) {
		threadPool.emplace_back([=] { this->partitionBasedColoring(n_cols, i, this->MAX_THREADS_SOLVE); });
	}

	for (auto& t : threadPool) {
		t.join();
	}
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
	this->partitionBasedColoring(n_cols, 0, 1);
#endif

	n_cols = *std::max_element(this->col.begin(), this->col.end()) + 1;

#ifdef COMPUTE_ELAPSED_TIME
	bm.sampleTimeToFlag(2);
#endif

	return n_cols;
}

int GebremedhinManne::performRecoloring(int n_cols) {
	for (int i = 0; i < this->recolor.size(); ++i) {
		int v = this->recolor[i];

		n_cols = this->computeVertexColor(v, n_cols, &this->col[v]);
	}

	return n_cols;
}

void GebremedhinManne::partitionBasedColoring(int n_cols, int const initial, int const displacement) {
	int const nV = this->adj().nV();
#if defined(COLORING_SYNCHRONOUS) && !defined(SEQUENTIAL_GRAPH_COLOR)
	int const nVCeil = (nV / displacement + 1) * displacement;
	for (int v = initial; v < nVCeil; v += displacement) {
		if (v < nV) {
			n_cols = this->computeVertexColor(v, n_cols, &this->col[v]);
		}
		this->barrier->wait();
	}
#endif
#if defined(COLORING_ASYNCHRONOUS) || defined(SEQUENTIAL_GRAPH_COLOR)
	for (int v = initial; v < nV; v += displacement) {
		n_cols = this->computeVertexColor(v, n_cols, &this->col[v]);
	}
#endif

#ifdef PARALLEL_GRAPH_COLOR
	if (initial == 0) {
#ifdef COMPUTE_ELAPSED_TIME
		Benchmark& bm = *Benchmark::getInstance();
		bm.sampleTimeToFlag(1);
#endif
	}

	this->barrier->wait();
	int step;
	for (int v = initial, step = 0; v < nV; v += displacement, ++step) {
		auto const end = this->adj().endNeighs(v);
		for (auto it = this->adj().beginNeighs(v); it != end; ++it) {
			int w = *it;
#ifdef COLORING_SYNCHRONOUS
			if (w / displacement != step) continue;
#endif
			if (v < w && this->col[v] == this->col[w]) {
				this->col[v] = GebremedhinManne::INVALID_COLOR;
				this->mutex.lock();
				this->recolor.push_back(v);
				this->mutex.unlock();
			}
		}
	}
#endif

	return;
}

const int GebremedhinManne::solve() {
	int n_cols = 0;

#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();
#endif

#ifdef SEQUENTIAL_GRAPH_COLOR
	n_cols = this->colorGraph(n_cols);
#ifdef COMPUTE_ELAPSED_TIME
	bm.sampleTimeToFlag(1);
#endif
#endif
#ifdef PARALLEL_GRAPH_COLOR
	n_cols = this->colorGraph(n_cols);
	++this->nIterations;
	this->nConflicts += this->recolor.size();
#ifdef COMPUTE_ELAPSED_TIME
	bm.sampleTimeToFlag(2);
#endif

	if (this->nConflicts > 0) {
		int index = 0;
		n_cols = this->performRecoloring(n_cols);

#ifdef COMPUTE_ELAPSED_TIME
		bm.sampleTimeToFlag(3);
#endif
		++this->nIterations;
	}
#endif

	return n_cols;
}

const int GebremedhinManne::getConflicts() const {
	return this->nConflicts;
}

const int GebremedhinManne::getIterations() const {
	return this->nIterations;
}
