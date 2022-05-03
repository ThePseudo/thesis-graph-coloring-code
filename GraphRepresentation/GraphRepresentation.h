#ifndef _GRAPH_REPRESENTATION_H
#define _GRAPH_REPRESENTATION_H

#include "configuration.h"

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
	virtual const int nV() const = 0;
	// Returns whether v is neighbor of w
	virtual const bool get(int v, int w) const = 0;

	const int countNeighs(int v) const;
	static void printGraphRepresentationConfs();

	// Returns begin iterator of neighbors of v
	virtual const ::std::vector<int>::const_iterator beginNeighs(int v) const = 0;
	// Returns end iterator of neighbors of v
	virtual const ::std::vector<int>::const_iterator endNeighs(int v) const = 0;

	void printGraphInfo() const;
};

#endif // !_GRAPH_REPRESENTATION_H
