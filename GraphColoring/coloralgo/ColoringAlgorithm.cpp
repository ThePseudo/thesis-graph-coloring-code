#include "ColoringAlgorithm.h"

#include <algorithm>
#include <map>

#include "benchmark.h"

int ColoringAlgorithm::INVALID_COLOR = -1;

void ColoringAlgorithm::printColorAlgorithmConfs() {
	std::cout << "Coloring Algorithm: ";
#ifdef COLORING_ALGORITHM_CUSPARSE
	std::cout << "Cohen-Castonguay (cuSPARSE)";
#else
#ifdef SEQUENTIAL_GRAPH_COLOR
	std::cout << "Sequential ";
#ifdef COLORING_ALGORITHM_GREEDY
	std::cout << "Greedy ";
#ifdef SORT_LARGEST_DEGREE_FIRST
	std::cout << "Largest Degree First ";
#endif
#ifdef SORT_SMALLEST_DEGREE_FIRST
	std::cout << "Smaller Degree First ";
#endif
#ifdef SORT_VERTEX_ORDER
	std::cout << " ";
#endif
#ifdef SORT_VERTEX_ORDER_REVERSED
	std::cout << "Reversed ";
#endif
#endif
#ifdef COLORING_ALGORITHM_JP
	std::cout << "Jones-Plassmann";
#endif
#ifdef COLORING_ALGORITHM_GM
#ifdef USE_IMPROVED_ALGORITHM
	std::cout << "Improved ";
#endif
	std::cout << "Gebremedhin-Manne ";
#ifdef COLORING_SYNCHRONOUS
	std::cout << "Sync";
#endif
#ifdef COLORING_ASYNCHRONOUS
	std::cout << "Async";
#endif
#endif
#endif
#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Parallel ";
#ifdef COLORING_ALGORITHM_GREEDY
	std::cout << "Greedy ";
#ifdef SORT_LARGEST_DEGREE_FIRST
	std::cout << "Largest Degree First ";
#endif
#ifdef SORT_SMALLEST_DEGREE_FIRST
	std::cout << "Smaller Degree First ";
#endif
#ifdef SORT_VERTEX_ORDER
	std::cout << " ";
#endif
#ifdef SORT_VERTEX_ORDER_REVERSED
	std::cout << "Reversed ";
#endif
#ifdef PARALLEL_RECOLOR
	std::cout << "with Parallel Recolor";
#endif
#ifdef SEQUENTIAL_RECOLOR
	std::cout << "with Sequential Recolor";
#endif
#endif
#ifdef COLORING_ALGORITHM_JP
	std::cout << "Jones-Plassmann";
#if defined(GRAPH_REPRESENTATION_CSR) && defined(PARALLEL_GRAPH_COLOR) && defined(USE_CUDA_ALGORITHM)
	std::cout << " (CUDA) ";
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
	std::cout << "Min-Max Independent Sets";
#endif
#endif
#endif
#ifdef COLORING_ALGORITHM_GM
#ifdef USE_IMPROVED_ALGORITHM
	std::cout << "Improved ";
#endif
	std::cout << "Gebremedhin-Manne ";
#ifdef COLORING_SYNCHRONOUS
	std::cout << "Sync";
#endif
#ifdef COLORING_ASYNCHRONOUS
	std::cout << "Async";
#endif
#endif
#endif
#endif

	std::cout << std::endl;
}

const GRAPH_REPR_T& ColoringAlgorithm::adj() const {
	return *this->_adj;
}

const std::vector<int> ColoringAlgorithm::getColors() const {
	return this->col;
}

void ColoringAlgorithm::init() {
	this->resetCount = -1;
	this->col = std::vector<int>(this->adj().nV());
}

void ColoringAlgorithm::reset() {
	++this->resetCount;

	std::fill(this->col.begin(), this->col.end(), this->INVALID_COLOR);
}

const int ColoringAlgorithm::computeVertexColor(int const v, int const n_cols, int* targetCol) const {
	int colorsNum = n_cols;
	auto neighIt = this->adj().beginNeighs(v);
	auto forbidden = std::vector<bool>(colorsNum, false);
	auto const end = this->adj().endNeighs(v);
	while (neighIt != end) {
		int w = *neighIt;
		int c = this->col[w];

		if (c != ColoringAlgorithm::INVALID_COLOR) {
#ifdef PARALLEL_GRAPH_COLOR
			if (c >= colorsNum) {
				colorsNum = c + 1;
				forbidden.resize(colorsNum, false);
			}
#endif
			forbidden[c] = true;
		}
		++neighIt;
	}
	auto targetIt = std::find(forbidden.begin(), forbidden.end(), false);
	//int targetCol;
	if (targetIt == forbidden.end()) {
		// All forbidden. Add new color
		*targetCol = colorsNum++;
	} else {
		*targetCol = targetIt - forbidden.begin();
	}

	return colorsNum;
}

std::vector<std::pair<int, int>> ColoringAlgorithm::checkCorrectColoring() {
	std::vector<std::pair<int, int>> incorrect;
	for (int v = 0; v < this->adj().nV(); ++v) {
		auto const end = this->adj().endNeighs(v);
		for (auto it = this->adj().beginNeighs(v); it != end; ++it) {
			int w = *it;
			if (v != w && this->col[v] == this->col[w]) {
				incorrect.push_back(std::pair<int, int>(v, w));
			}
		}
	}

	return incorrect;
}

void ColoringAlgorithm::printBenchmarkInfo() const {
	std::cout << "TIME USAGE" << std::endl;
	this->adj().printBenchmarkInfo();
}

void ColoringAlgorithm::printColors(std::ostream& os) const {
	for (int v = 0; v < this->adj().nV(); ++v) {
		os << v << ": " << this->col[v] << std::endl;
	}
}

void ColoringAlgorithm::printHisto(std::ostream& os) const {
	std::map<int, int, std::less<int>> colorHisto;

	for (const auto& color : this->col) {
		if (colorHisto.find(color) == colorHisto.end()) {
			colorHisto.insert(std::pair<int, int>(color, 0));
		}
		colorHisto[color] += 1;
	}

	for (const auto& histoCol : colorHisto) {
		os << histoCol.first << ": " << histoCol.second << std::endl;
	}
}

void ColoringAlgorithm::printDotFile(std::ostream& os) const {
	os << "strict graph {" << std::endl;
	os << "\tnode [colorscheme=pastel19]" << std::endl;
	// Write vertexes
	for (int v = 0; v < this->adj().nV(); ++v) {
		os << "\t" << v << "[style=filled, color=" << this->col[v] + 1 << "]" << std::endl;
	}
	// Write edges
	for (int v = 0; v < this->adj().nV(); ++v) {
		auto adjIt = this->adj().beginNeighs(v);
		auto const end = this->adj().endNeighs(v);
		while (adjIt != end) {
			int w = *adjIt;
			if (v <= w) {
				os << "\t" << v << " -- " << w << std::endl;
			}
			++adjIt;
		}
	}
	os << "}" << std::endl;
}
