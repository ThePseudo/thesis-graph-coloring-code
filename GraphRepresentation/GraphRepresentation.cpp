#include "GraphRepresentation.h"

const int GraphRepresentation::countNeighs(size_t v) const {
	if (0 > v || v >= this->nV()) {
		throw std::out_of_range("v index out of range: " + v);
	}

	return this->endNeighs(v) - this->beginNeighs(v);
}