#include "JonesPlassmann.h"

#include "benchmark.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <random>

JonesPlassmann::JonesPlassmann(std::string const filepath) {
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

#ifdef PARALLEL_GRAPH_COLOR
	this->barrier = nullptr;
#endif
	this->nIterations = 0;
}

void JonesPlassmann::init() {
	__super::init();

	this->vWeights = std::vector<int>(this->adj().nV());
#ifdef SEQUENTIAL_GRAPH_COLOR
	this->nWaits = std::vector<int>(this->adj().nV());
#elif defined(PARALLEL_GRAPH_COLOR)
	this->MAX_THREADS_SOLVE = std::min(this->adj().nV(), this->MAX_THREADS_SOLVE);
	this->barrier = new Barrier(this->MAX_THREADS_SOLVE);
	this->nWaits = std::vector<std::atomic_int>(this->adj().nV());
	this->firstAndLasts = std::vector<std::pair<int, int>>(this->MAX_THREADS_SOLVE);
	this->partitionVertices();
	this->n_colors = std::vector<int>(this->MAX_THREADS_SOLVE);
#endif
}

void JonesPlassmann::reset() {
	__super::reset();

	Benchmark& bm = *Benchmark::getInstance(__super::resetCount);
	bm.sampleTime();

	srand(static_cast<unsigned>(time(0)));

	for (int i = 0; i < this->adj().nV(); ++i) {
		this->vWeights[i] = i;
		this->nWaits[i] = 0;
	}
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(this->vWeights.begin(), this->vWeights.end(), g);

	this->nIterations = 0;

	bm.sampleTimeToFlag(1);
}

const int JonesPlassmann::startColoring() {
#if defined(COLORING_ALGORITHM_JP) && defined(GRAPH_REPRESENTATION_CSR) && defined(PARALLEL_GRAPH_COLOR) && defined(USE_CUDA_ALGORITHM)
	return this->colorWithCuda();
#else
	return this->solve();
#endif
}

const int JonesPlassmann::solve() {
	Benchmark& bm = *Benchmark::getInstance(__super::resetCount);
	int n_cols = 0;

	bm.sampleTime();

#ifdef SEQUENTIAL_GRAPH_COLOR
	this->coloringHeuristic(0, this->adj().nV(), n_cols);
#endif
#ifdef PARALLEL_GRAPH_COLOR
	std::vector<std::future<void>> threadPool;
	threadPool.reserve(this->MAX_THREADS_SOLVE);
	bm.sampleTimeToFlag(1);
	
	for (int i = 0; i < this->MAX_THREADS_SOLVE; ++i) {
		auto& firstAndLast = this->firstAndLasts[i];
		auto& ncols = this->n_colors[i];
		threadPool.push_back(
			std::async(std::launch::async, &JonesPlassmann::coloringHeuristic, this, firstAndLast.first, firstAndLast.second, std::ref(n_cols))
		);
	}

	for (auto& t : threadPool) {
		t.get();
	}

	n_cols = *std::max_element(this->n_colors.begin(), this->n_colors.end());
#endif

	bm.sampleTimeToFlag(2);

	return n_cols;
}

const int JonesPlassmann::getIterations() const {
	return this->nIterations;
}

#ifdef PARALLEL_GRAPH_COLOR
void JonesPlassmann::partitionVertices() {
#ifdef PARTITION_VERTICES_EQUALLY
	this->partitionVerticesEqually();
#endif
#ifdef PARTITION_VERTICES_BY_EDGE_NUM
	this->partitionVerticesByEdges();
#endif
}

void JonesPlassmann::partitionVerticesEqually() {
	int const nThreads = this->MAX_THREADS_SOLVE;
	int const nV = this->adj().nV();
	int const thresh = nV / nThreads;
	int first = 0;
	int last = thresh;
	

	for (int i = 0; i < nThreads - 1; ++i) {
		this->firstAndLasts[i] = std::pair<int, int>(first, last);
		first = last;
		last += thresh;
	}
	this->firstAndLasts[nThreads - 1] = std::pair<int, int>(first, nV);
}

void JonesPlassmann::partitionVerticesByEdges() {
	int first = 0;
	int last = 0;
	size_t acc = 0;
	int const nThreads = this->MAX_THREADS_SOLVE;
	size_t const thresh = this->adj().nE() / nThreads;
	int const nV = this->adj().nV();

	for (int i = 0; i < nThreads; ++i) {
		while (i != nThreads - 1 && last < nV && acc < thresh) {
			acc += this->adj().countNeighs(last);
			++last;
		}
		while (i == nThreads - 1 && last < nV) {
			acc += this->adj().countNeighs(last);
			++last;
		}
		if (i + 1 < nThreads) ++last;
		this->firstAndLasts[i] = std::pair<int, int>(first, last);
		first = last;
		acc = 0;
	}
}
#endif

void JonesPlassmann::coloringHeuristic(int const first, int const last, int& n_cols) {
	this->calcWaitTime(first, last);
#ifdef PARALLEL_GRAPH_COLOR
	this->barrier->wait();
#endif
	this->colorWhileWaiting(first, last, n_cols);
}

void JonesPlassmann::calcWaitTime(int const first, int const last) {
	for (int v = first; v < this->adj().nV() && v < last; ++v) {
		auto const end = this->adj().endNeighs(v);
		for (auto neighIt = this->adj().beginNeighs(v);
			neighIt != end; ++neighIt) {
			int w = *neighIt;
			
			// Ordering by random weight
			if (this->vWeights[w] > this->vWeights[v]) {
				++this->nWaits[v];
			}
		}
	}
}

void JonesPlassmann::colorWhileWaiting(int const first, int const last, int& n_cols) {
	bool again = true;
	int const nV = this->adj().nV();
	do {
		again = false;
		for (int v = first; v < nV && v < last; ++v) {
			if (this->nWaits[v] == 0) {
				n_cols = this->computeVertexColor(v, n_cols, &this->col[v]);
				--this->nWaits[v];
				auto const end = this->adj().endNeighs(v);
				for (auto neighIt = this->adj().beginNeighs(v); neighIt != end; ++neighIt) {
					--this->nWaits[*neighIt];
				}
			} else if (this->nWaits[v] > 0) {
				again = true;
			}
		}

		if (first == 0) {
			++this->nIterations;
		}
	} while (again);
}

#if defined(COLORING_ALGORITHM_JP) && defined(GRAPH_REPRESENTATION_CSR) && defined(PARALLEL_GRAPH_COLOR) && defined(USE_CUDA_ALGORITHM)
#include "cudaKernels.h"
const int JonesPlassmann::colorWithCuda() {
	int const n = this->adj().nV();
	const int* Ao = this->adj().getRowPointers();
	const int* Ac = this->adj().getColIndexes();
	int* colors = this->col.data();
	const int* randoms = this->vWeights.data();

	//size_t total_size = (n + 1) * sizeof(*Ao) + Ao[n] * sizeof(Ac) + n * sizeof(*colors) + n * sizeof(*randoms);
	//std::cout << "Transfering " << total_size << " bytes to GPU memory." << std::endl;	

	this->nIterations = color_jpl(n, Ao, Ac, colors, randoms, __super::resetCount);

#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
	return *std::max_element(this->col.begin(), this->col.end()) + 1;
#endif
#ifdef COLOR_MAX_INDEPENDENT_SET
	return this->nIterations;
#endif
}
#endif

void JonesPlassmann::printExecutionInfo() const {
	std::cout << "Solution converged to in " << this->getIterations() << " iterations." << std::endl;
}

void JonesPlassmann::printBenchmarkInfo() const {
	__super::printBenchmarkInfo();

	Benchmark& bm = *Benchmark::getInstance(__super::resetCount);
#if defined(GRAPH_REPRESENTATION_CSR) && defined(PARALLEL_GRAPH_COLOR) && defined(USE_CUDA_ALGORITHM)
	std::cout << "TXfer to GPU:\t\t" << bm.getTimeOfFlag(2) << " s" << std::endl;
#endif
	std::cout << "Vertex color:\t\t" << bm.getTimeOfFlag(1) << " s" << std::endl;
#if defined(GRAPH_REPRESENTATION_CSR) && defined(PARALLEL_GRAPH_COLOR) && defined(USE_CUDA_ALGORITHM)
	std::cout << "TXfer from GPU:\t\t" << bm.getTimeOfFlag(3) << " s" << std::endl;
	std::cout << "Count left:\t\t" << bm.getTimeOfFlag(4) << " s" << std::endl;
#endif

	std::cout << "Total:\t\t" << bm.getTotalTime() << " s" << std::endl;
}