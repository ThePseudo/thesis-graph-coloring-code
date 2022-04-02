#include "JonesPlassmann.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include <algorithm>
#include <fstream>
#include <iterator>
#include <set>

JonesPlassmann::JonesPlassmann(std::string const filepath) {
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

	this->col = std::vector<int>(this->adj().nV());
	this->vWeights = std::vector<float>(this->adj().nV());

	//srand(static_cast<unsigned>(time(0)));
	// TODO: Remove static random seed
	srand(static_cast<unsigned>(121));

	for (int i = 0; i < this->adj().nV(); ++i) {
		this->col[i] = this->INVALID_COLOR;
		this->vWeights[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);;
	}

	this->nIterations = 0;
}

const int JonesPlassmann::startColoring() {
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(1);
	bm.clear(2);

	return this->solve();
}

const int JonesPlassmann::solve() {
	std::set<size_t> toAnalyze;
	std::set<size_t> independent;
	std::set<size_t> diff;
	for (size_t i = 0; i < this->adj().nV(); ++i) {
		toAnalyze.insert(i);
	}

	int n_cols = 0;

#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();
#endif

	while (!toAnalyze.empty()) {
		// Find independent set of verteces not yet colored
		independent.clear();
		auto it = toAnalyze.begin();
#ifdef SEQUENTIAL_GRAPH_COLOR
		this->findIndependentSet(it, toAnalyze.end(), independent);
#endif
#ifdef PARALLEL_GRAPH_COLOR
		std::vector<std::thread> threadPool;

		for (int i = 0; i < this->MAX_THREADS_SOLVE; ++i) {
			threadPool.emplace_back([&] { this->findIndependentSet(it, toAnalyze.end(), independent); });
		}

		for (auto& t : threadPool) {
			t.join();
		}
#endif

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

	return n_cols;
}

void JonesPlassmann::findIndependentSet(std::set<size_t>::iterator& first, std::set<size_t>::iterator last, std::set<size_t>& indSet) {
	while (true) {
#ifdef PARALLEL_GRAPH_COLOR
		this->mutex.lock();
#endif
		if (first == last) {
#ifdef PARALLEL_GRAPH_COLOR
			this->mutex.unlock();
#endif
			break;
		}
		auto& v = *first;
		++first;
#ifdef PARALLEL_GRAPH_COLOR
		this->mutex.unlock();
#endif

		float selfWeight = this->vWeights[v];
		bool isMax = true;
		// FIX: Iterate on toAnalyze `intersect` neighbors
		for (auto it = this->adj().beginNeighs(v);
			isMax && it < this->adj().endNeighs(v); ++it) {

#ifdef PARALLEL_GRAPH_COLOR
			this->mutex.lock();
#endif
			float neighWeight = this->vWeights[*it];
#ifdef PARALLEL_GRAPH_COLOR
			this->mutex.unlock();
#endif
			if (selfWeight < neighWeight) {
				isMax = false;
				break;
			}
		}

		if (isMax) {
#ifdef PARALLEL_GRAPH_COLOR
			this->mutex.lock();
#endif
			indSet.insert(v);
#ifdef PARALLEL_GRAPH_COLOR
			this->mutex.unlock();
#endif
			this->vWeights[v] = 0.0f;
		}
	}
}

const int JonesPlassmann::getIterations() const {
	return this->nIterations;
}