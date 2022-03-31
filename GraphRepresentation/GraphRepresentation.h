#ifndef _GRAPH_REPRESENTATION_H
#define _GRAPH_REPRESENTATION_H

#include <iostream>
#include <istream>
#include <vector>

class GraphRepresentation {
public:
	// Read graph from input stream
	friend std::istream& operator>>(std::istream& is, GraphRepresentation& m) { return is; };

	// Returns total number of edges
	virtual const size_t nE() const = 0;
	// Returns total number of verteces
	virtual const size_t nV() const = 0;
	// Returns whether v is neighbor of w
	virtual const bool get(size_t v, size_t w) const = 0;

	const int countNeighs(size_t v) const;

	// Returns begin iterator of neighbors of v
	virtual const ::std::vector<size_t>::const_iterator beginNeighs(size_t v) const = 0;
	// Returns end iterator of neighbors of v
	virtual const ::std::vector<size_t>::const_iterator endNeighs(size_t v) const = 0;
};

#endif // !_GRAPH_REPRESENTATION_H
