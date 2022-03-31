#include "JonesPlassmann.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include <algorithm>
#include <iterator>
#include <set>

JonesPlassmann::JonesPlassmann(GraphRepresentation& gr) {
	this->_adj = &gr;
	this->col = std::vector<int>(this->adj().nV());
	this->vWeights = std::vector<float>(this->adj().nV());

	//srand(static_cast<unsigned>(time(0)));
	// TODO: Remove static random seed
	srand(static_cast<unsigned>(121));

	for (int i = 0; i < this->adj().nV(); ++i) {
		this->col[i] = JonesPlassmann::INVALID_COLOR;
		this->vWeights[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);;
	}
}

const GraphRepresentation& JonesPlassmann::adj() {
	return *this->_adj;
}

const std::vector<int> JonesPlassmann::getColors() {
	return this->col;
}

const int JonesPlassmann::solve() {
	std::set<size_t> toAnalyze;
	std::set<size_t> indipendent;
	std::set<size_t> diff;
	for (size_t i = 0; i < this->adj().nV(); ++i) {
		toAnalyze.insert(i);
	}

	int n_cols = 0;

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

	while (!toAnalyze.empty()) {
		indipendent.clear();
		for (auto& v : toAnalyze) {
			float selfWeight = this->vWeights[v];
			bool isMax = true;
			for (auto it = this->adj().beginNeighs(v);
				isMax && it < this->adj().endNeighs(v); ++it) {

				if (selfWeight < this->vWeights[*it]) {
					isMax = false;
					break;
				}
			}

			if (isMax) {
				indipendent.insert(v);
				this->vWeights[v] = 0.0f;
			}
		}

#ifdef COMPUTE_ELAPSED_TIME
		sampleTime();
		sortTime += getElapsedTime();
#endif
		for (auto& v : indipendent) {
			auto neighIt = this->adj().beginNeighs(v);
			auto forbidden = std::vector<bool>(n_cols);
			std::fill(forbidden.begin(), forbidden.end(), false);
			while (neighIt != this->adj().endNeighs(v)) {
				int w = *neighIt;
				int c = this->col[w];

				if (c != JonesPlassmann::INVALID_COLOR) forbidden[c] = true;
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

#ifdef COMPUTE_ELAPSED_TIME
		sampleTime();
		colorTime += getElapsedTime();
#endif

		std::set_difference(toAnalyze.begin(), toAnalyze.end(),
			indipendent.begin(), indipendent.end(),
			std::inserter(diff, diff.begin()));

		toAnalyze.clear();
		toAnalyze = diff;
		diff.clear();
	}

	return n_cols;
}