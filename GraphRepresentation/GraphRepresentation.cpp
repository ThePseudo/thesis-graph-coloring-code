#include "GraphRepresentation.h"

const int GraphRepresentation::countNeighs(int v) const {
	if (0 > v || v >= this->nV()) {
		throw std::out_of_range("v index out of range: " + v);
	}

	return this->endNeighs(v) - this->beginNeighs(v);
}

void GraphRepresentation::printGraphRepresentationConfs() {
	std::cout << "Graph Representation: ";
#ifdef GRAPH_REPRESENTATION_ADJ_MATRIX
	std::cout << "Adjacency Matrix";
#endif
#ifdef GRAPH_REPRESENTATION_CSR
	std::cout << "Compressed Sparse Row";
#endif
	std::cout << std::endl;
}

void GraphRepresentation::printGraphInfo() const {
	std::cout << "V: " << this->nV() << ", E: " << this->nE() << std::endl;
}