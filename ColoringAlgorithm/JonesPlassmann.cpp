#include "JonesPlassmann.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include <algorithm>
#include <fstream>
#include <iterator>
#include <set>

JonesPlassmann::JonesPlassmann(std::string const filepath) {
#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(0);
#endif

	this->_adj = new GRAPH_REPR_T();

	std::ifstream fileIS;
	fileIS.open(filepath);
	std::istream& is = fileIS;
	is >> *this->_adj;

	if (fileIS.is_open()) {
		fileIS.close();
	}

	this->col = std::vector<int>(this->adj().nV());
	this->vWeights = std::vector<float>(this->adj().nV());
	
	this->nWaits = std::vector<std::atomic_int>(this->adj().nV());
	this->firstAndLasts = std::vector<std::pair<size_t, size_t>>(this->MAX_THREADS_SOLVE);

	srand(static_cast<unsigned>(time(0)));
	// TODO: Remove static random seed
	//srand(static_cast<unsigned>(121));

	for (int i = 0; i < this->adj().nV(); ++i) {
		this->col[i] = this->INVALID_COLOR;
		this->vWeights[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}

	this->nIterations = 0;
}

const int JonesPlassmann::startColoring() {
#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(1);
	bm.clear(2);
#endif

	return this->solve();
}

const int JonesPlassmann::solve() {
	int n_cols = 0;

#ifdef SEQUENTIAL_GRAPH_COLOR
#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();
#endif
	std::set<size_t> toAnalyze;
	std::set<size_t> independent;
	std::set<size_t> diff;
	for (size_t i = 0; i < this->adj().nV(); ++i) {
		toAnalyze.insert(i);
	}
	while (!toAnalyze.empty()) {
		// Find independent set of verteces not yet colored
		independent.clear();
		auto it = toAnalyze.begin();

		this->findIndependentSet(it, toAnalyze.end(), independent);
#ifdef COMPUTE_ELAPSED_TIME
		bm.sampleTimeToFlag(1);
#endif
		for (auto& v : independent) {
			n_cols = this->computeVertexColor(
				v, n_cols, &this->col[v]
			);
		}

#ifdef COMPUTE_ELAPSED_TIME
		bm.sampleTimeToFlag(2);
#endif

		std::set_difference(toAnalyze.begin(), toAnalyze.end(),
			independent.begin(), independent.end(),
			std::inserter(diff, diff.begin()));

		toAnalyze.clear();
		toAnalyze = diff;
		diff.clear();

		++this->nIterations;
	}
#endif

#ifdef PARALLEL_GRAPH_COLOR
#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();
#endif
	bool again = true;
	std::vector<std::thread> threadPool;
	int const nThreads = std::min(this->adj().nV(), static_cast<size_t>(this->MAX_THREADS_SOLVE));
	
	this->barrier = new Barrier(nThreads);
	this->partitionVerticesByEdges(nThreads);
	
	threadPool.reserve(nThreads);
	for (int i = 0; i < nThreads; ++i) {
		auto& firstAndLast = this->firstAndLasts[i];
		threadPool.emplace_back(
			[=, &n_cols, &again] { this->coloringHeuristic(firstAndLast.first, firstAndLast.second, n_cols, again); }
		);
	}

	for (auto& t : threadPool) {
		t.join();
	}

#ifdef COMPUTE_ELAPSED_TIME
	bm.sampleTimeToFlag(1);
#endif
#endif

	return n_cols;
}

void JonesPlassmann::findIndependentSet(std::set<size_t>::iterator& first, std::set<size_t>::iterator last, std::set<size_t>& indSet) {
	while (true) {
		if (first == last) {
			break;
		}
		auto& v = *first;
		++first;
		float selfWeight = this->vWeights[v];
		bool isMax = true;
		// FIX: Iterate on toAnalyze `intersect` neighbors
		for (auto it = this->adj().beginNeighs(v);
			isMax && it < this->adj().endNeighs(v); ++it) {
			float neighWeight = this->vWeights[*it];
			if (selfWeight < neighWeight) {
				isMax = false;
				break;
			}
		}

		if (isMax) {
			indSet.insert(v);
			this->vWeights[v] = 0.0f;
		}
	}
}

const int JonesPlassmann::getIterations() const {
	return this->nIterations;
}

void JonesPlassmann::partitionVerticesByEdges(int const nThreads) {
	size_t first = 0;
	size_t last = 0;
	size_t acc = 0;
	size_t thresh = this->adj().nE() / nThreads;

	for (int i = 0; i < nThreads; ++i) {
		while (i != nThreads - 1 && last < this->adj().nV() && acc < thresh) {
			acc += this->adj().countNeighs(last);
			++last;
		}
		while (i == nThreads - 1 && last < this->adj().nV()) {
			acc += this->adj().countNeighs(last);
			++last;
		}
		if (i + 1 < nThreads) ++last;
		this->firstAndLasts[i] = std::pair<size_t, size_t>(first, last);
		first = last;
		acc = 0;
	}
}

void JonesPlassmann::coloringHeuristic(size_t const first, size_t const last, int& n_cols, bool& again) {
	this->calcWaitTime(first, last);
	this->barrier->wait();
	this->colorWhileWaiting(first, last, n_cols, again);
}

void JonesPlassmann::calcWaitTime(size_t const first, size_t const last) {
	for (size_t v = first; v < this->adj().nV() && v < last; ++v) {
		for (auto neighIt = this->adj().beginNeighs(v);
			neighIt != this->adj().endNeighs(v); ++neighIt) {
			size_t w = *neighIt;
			
			// Ordering by random weight > n neighbors > index
			if (this->vWeights[w] > this->vWeights[v]) {
				++this->nWaits[v];
			} else if (this->vWeights[w] == this->vWeights[v]) {
				if (this->adj().countNeighs(w) > this->adj().countNeighs(v)) {
					++this->nWaits[v];
				} else if (this->adj().countNeighs(w) == this->adj().countNeighs(v)) {
					if (w > v) {
						++this->nWaits[v];
					}
				}
			}

			// Branchless conditional
			//this->nWaits[v] += 
			//	(this->vWeights[w] >  this->vWeights[v]) +
			//	((this->vWeights[w] == this->vWeights[v]) * (this->adj().countNeighs(w) >  this->adj().countNeighs(v))) +
			//	((this->vWeights[w] == this->vWeights[v]) * (this->adj().countNeighs(w) == this->adj().countNeighs(v)) * (w > v));
		}
	}
}

void JonesPlassmann::colorWhileWaiting(size_t const first, size_t const last, int& n_cols, bool& again) {
	int nColors = n_cols;
	while (again) {
		this->barrier->wait();
		again = false;
		this->barrier->wait();
		for (size_t v = first; v < this->adj().nV() && v < last; ++v) {
			if (this->nWaits[v] == 0) {
				nColors = this->computeVertexColor(v, nColors, &this->col[v]);
				--this->nWaits[v];
				for (auto neighIt = this->adj().beginNeighs(v); neighIt != this->adj().endNeighs(v); ++neighIt) {
					--this->nWaits[*neighIt];
				}
			} else if (!again && this->nWaits[v] > 0) {
				again = true;
			}
		}

		this->barrier->wait();
		if (first == 0) {
			++this->nIterations;
		}
	}

	this->mutex.lock();
	if (nColors > n_cols) {
		n_cols = nColors;
	}
	this->mutex.unlock();
}