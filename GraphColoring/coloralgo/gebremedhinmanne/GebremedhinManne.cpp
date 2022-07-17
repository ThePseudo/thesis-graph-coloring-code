//#include "configuration.h"
#include "GebremedhinManne.h"
#include "benchmark.h"

#include <algorithm>
#include <fstream>

GebremedhinManne::GebremedhinManne(std::string const filepath) {
	Benchmark& bm = *Benchmark::getInstance(0);

	this->_adj = new GRAPH_REPR_T();

	std::ifstream fileIS;
	fileIS.open(filepath);
	std::istream& is = fileIS;

	bm.sampleTime();
	is >> *this->_adj;
	bm.sampleTimeToFlag(0);

	if (fileIS.is_open()) {
		fileIS.close();
	}

	this->barrier = nullptr;
	this->nConflicts = 0;
	this->nIterations = 0;
}

void GebremedhinManne::init() {
	__super::init();

	this->MAX_THREADS_SOLVE = std::min(this->adj().nV(), this->MAX_THREADS_SOLVE);

	// Pre-allocate vector to avoid reallocation at runtime
	// Size is given by Lemma 1 of Gebremedhin-Manne paper.
	int const maxConflicts = 1.0f * this->adj().avgD() * (this->MAX_THREADS_SOLVE - 1) / 2 * this->adj().nV() / (this->adj().nV() - 1);
	this->recolor.reserve(maxConflicts);

	this->barrier = new Barrier(this->MAX_THREADS_SOLVE);

#ifdef USE_IMPROVED_ALGORITHM
	this->colorClasses = std::unordered_map<int, std::vector<int>>();
#endif
}

void GebremedhinManne::reset() {
	__super::reset();

	Benchmark& bm = *Benchmark::getInstance(__super::resetCount);
	bm.sampleTime();

	this->nConflicts = 0;
	this->nIterations = 0;

	this->recolor.clear();

#ifdef USE_IMPROVED_ALGORITHM
	this->colorClasses.clear();
#endif

	bm.sampleTimeToFlag(1);
}

const int GebremedhinManne::startColoring() {
	return this->solve();
}

const int GebremedhinManne::solve() {
	Benchmark& bm = *Benchmark::getInstance(__super::resetCount);
	int n_cols = 0;
	
	bm.sampleTime();

#ifdef SEQUENTIAL_GRAPH_COLOR
	n_cols = this->colorGraph(n_cols);
#endif
#ifdef PARALLEL_GRAPH_COLOR
	n_cols = this->colorGraph(n_cols);
	++this->nIterations;
	this->nConflicts += this->recolor.size();

	if (this->nConflicts > 0) {
		int index = 0;
		// Phase 3
		n_cols = this->performRecoloring(n_cols);
		++this->nIterations;
	}
#endif

	bm.sampleTimeToFlag(2);

	return n_cols;
}

int GebremedhinManne::colorGraph(int n_cols) {
#ifdef PARALLEL_GRAPH_COLOR
	std::vector<std::thread> threadPool;
	for (int i = 0; i < this->MAX_THREADS_SOLVE; ++i) {
#ifdef USE_STANDARD_ALGORITHM
		threadPool.emplace_back([=] { this->partitionBasedColoring(n_cols, i, this->MAX_THREADS_SOLVE); });
#endif
#ifdef USE_IMPROVED_ALGORITHM
		threadPool.emplace_back([=] { this->improvedPartitionBasedColoring(n_cols, i, this->MAX_THREADS_SOLVE); });
#endif
	}

	for (auto& t : threadPool) {
		t.join();
	}
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
#ifdef USE_STANDARD_ALGORITHM
	this->partitionBasedColoring(n_cols, 0, 1);
#endif
#ifdef USE_IMPROVED_ALGORITHM
	this->improvedPartitionBasedColoring(n_cols, 0, 1);
#endif
#endif

	n_cols = *std::max_element(this->col.begin(), this->col.end()) + 1;

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
	// Phase 1
#if defined(COLORING_SYNCHRONOUS) && !defined(SEQUENTIAL_GRAPH_COLOR)
	// The upper bound of the loop must be the minimun multiple of displacement greater than nV
	//  to account for the wait on the barrier.
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
	this->barrier->wait();
	int step;
	// Phase 2
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

void GebremedhinManne::improvedPartitionBasedColoring(int n_cols, int const initial, int const displacement) {
	int const nV = this->adj().nV();
	// Phase 1
#if defined(COLORING_SYNCHRONOUS) && !defined(SEQUENTIAL_GRAPH_COLOR)
	// The upper bound of the loop must be the minimun multiple of displacement greater than nV
	//  to account for the wait on the barrier.
	int const nVCeil = (nV / displacement + 1) * displacement;
	for (int v = initial; v < nVCeil; v += displacement) {
		if (v < nV) {
			n_cols = this->computeVertexColor(v, n_cols, &this->col[v]);
#ifdef PARALLEL_GRAPH_COLOR
			this->mutex.lock();
#endif
			this->colorClasses[this->col[v]].push_back(v);
#ifdef PARALLEL_GRAPH_COLOR
			this->mutex.unlock();
#endif
		}
#ifdef PARALLEL_GRAPH_COLOR
		this->barrier->wait();
#endif
	}
#endif
#if defined(COLORING_ASYNCHRONOUS) || defined(SEQUENTIAL_GRAPH_COLOR)
	for (int v = initial; v < nV; v += displacement) {
		n_cols = this->computeVertexColor(v, n_cols, &this->col[v]);
#ifdef PARALLEL_GRAPH_COLOR
		this->mutex.lock();
#endif
		this->colorClasses[this->col[v]].push_back(v);
#ifdef PARALLEL_GRAPH_COLOR
		this->mutex.unlock();
#endif
	}
#endif

	// Uncolor all nodes
#ifdef PARALLEL_GRAPH_COLOR
	this->barrier->wait();
#endif
	for (int v = initial; v < nV; v += displacement) {
		this->col[v] = GebremedhinManne::INVALID_COLOR;
	}

#ifdef PARALLEL_GRAPH_COLOR
	this->barrier->wait();
#endif
	// Phase 2
	for (int k = this->colorClasses.size() - 1; k >= 0; --k) {
		// Uncolor color class nodes
		//for (int vIdx = initial; vIdx < this->colorClasses[k].size(); vIdx += displacement) {
		//	int v = this->colorClasses[k][vIdx];
		//	this->col[v] = GebremedhinManne::INVALID_COLOR;
		//}
		//this->barrier->wait();
		for (int vIdx = initial; vIdx < this->colorClasses.at(k).size(); vIdx += displacement) {
			int v = this->colorClasses.at(k)[vIdx];
			n_cols = this->computeVertexColor(v, n_cols, &this->col[v]);
		}

		// Synchronize thread for every color class
#ifdef COLORING_SYNCHRONOUS
		this->barrier->wait();
#endif
	}
#ifdef PARALLEL_GRAPH_COLOR
	this->barrier->wait();
	// Phase 3
	for (int v = initial; v < nV; v += displacement) {
		auto const end = this->adj().endNeighs(v);
		for (auto it = this->adj().beginNeighs(v); it != end; ++it) {
			int w = *it;
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

const int GebremedhinManne::getConflicts() const {
	return this->nConflicts;
}

const int GebremedhinManne::getIterations() const {
	return this->nIterations;
}

void GebremedhinManne::printExecutionInfo() const {
#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Solution converged to in " << this->getIterations() << " iterations." << std::endl;
	std::cout << "Detected a total of " << this->getConflicts() << " conflicts." << std::endl;
#endif

	return;
}

void GebremedhinManne::printBenchmarkInfo() const {
	__super::printBenchmarkInfo();

	Benchmark& bm = *Benchmark::getInstance(__super::resetCount);
	std::cout << "Vertex color:\t\t" << bm.getTimeOfFlag(2) << " s" << std::endl;
#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Conflict search:\t" << bm.getTimeOfFlag(2) << " s" << std::endl;
	std::cout << "Vertex recolor:\t\t" << bm.getTimeOfFlag(3) << " s" << std::endl;
#endif

	std::cout << "Total:\t\t" << bm.getTotalTime() << " s" << std::endl;
}
