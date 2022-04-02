#include "ColoringAlgorithm.h"

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