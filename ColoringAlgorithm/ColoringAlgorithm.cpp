#include "ColoringAlgorithm.h"

void ColoringAlgorithm::printColorAlgorithmConfs() {
	std::cout << "Coloring Algorithm: ";
#ifdef SEQUENTIAL_GRAPH_COLOR
	std::cout << "Sequential ";
#ifdef COLORING_ALGORITHM_JP
	std::cout << "Jones-Plassmann";
#endif
#ifdef COLORING_ALGORITHM_GM
	std::cout << "Gebremedhin-Manne ";
#ifdef SORT_LARGEST_DEGREE_FIRST
	std::cout << "Largest Degree First";
#endif
#ifdef SORT_SMALLEST_DEGREE_FIRST
	std::cout << "Smaller Degree First";
#endif
#ifdef SORT_VERTEX_ORDER
	std::cout << "";
#endif
#ifdef SORT_VERTEX_ORDER_REVERSED
	std::cout << "Reversed";
#endif
#endif
#endif
#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Parallel ";
#ifdef COLORING_ALGORITHM_JP
	std::cout << "Jones-Plassmann";
#endif
#ifdef COLORING_ALGORITHM_GM
	std::cout << "Gebremedhin-Manne ";
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
#endif

	std::cout << std::endl;
}

const GraphRepresentation& ColoringAlgorithm::adj() const {
	return *this->_adj;
}

const std::vector<int> ColoringAlgorithm::getColors() const {
	return this->col;
}

const int ColoringAlgorithm::computeVertexColor(size_t const v, int const n_cols, int* targetCol) const {
	int colorsNum = n_cols;
	auto neighIt = this->adj().beginNeighs(v);
	auto forbidden = std::vector<bool>(colorsNum);
	std::fill(forbidden.begin(), forbidden.end(), false);
	while (neighIt != this->adj().endNeighs(v)) {
		size_t w = *neighIt;
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

std::vector<std::pair<size_t, size_t>> ColoringAlgorithm::checkCorrectColoring() {
	std::vector<std::pair<size_t, size_t>> incorrect;
	for (size_t v = 0; v < this->adj().nV(); ++v) {
		for (auto it = this->adj().beginNeighs(v); it < this->adj().endNeighs(v); ++it) {
			size_t w = *it;
			if (v != w && this->col[v] == this->col[w]) {
				incorrect.push_back(std::pair<size_t, size_t>(v, w));
			}
		}
	}

	return incorrect;
}

void ColoringAlgorithm::printColors(std::ostream& os) const {
	for (int v = 0; v < this->adj().nV(); ++v) {
		os << v << ": " << this->col[v] << std::endl;
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
	for (size_t v = 0; v < this->adj().nV(); ++v) {
		auto adjIt = this->adj().beginNeighs(v);
		while (adjIt != this->adj().endNeighs(v)) {
			size_t w = *adjIt;
			if (v <= w) {
				os << "\t" << v << " -- " << w << std::endl;
			}
			++adjIt;
		}
	}
	os << "}" << std::endl;
}