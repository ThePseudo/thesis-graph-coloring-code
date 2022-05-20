#ifndef _GRAPH_REPRESENTATION_H
#define _GRAPH_REPRESENTATION_H

//#include "configuration.h"

#include <iostream>
#include <istream>
#include <vector>

class GraphRepresentation {
protected:
	size_t _nE = 0;
	int _nV = 0;
	int _maxD = 0;
	int _minD = 0;
	float _avgD = 0.0;

public:
	// Read graph from input stream
	friend std::istream& operator>>(std::istream& is, GraphRepresentation& m) { return is; };

	// Returns total number of edges
	size_t nE() const;
	// Returns total number of verteces
	int nV() const;
	// Returns the maximum degree of the graph
	int maxD() const;
	// Returns the minimum degree of the graph
	int minD() const;
	// Returns the average degree of the graph
	float avgD() const;
	// Returns whether v is neighbor of w
	virtual bool get(int v, int w) const = 0;

	int countNeighs(int v) const;
	static void printGraphRepresentationConfs();

	// Returns begin iterator of neighbors of v
	virtual const ::std::vector<int>::const_iterator beginNeighs(int v) const = 0;
	// Returns end iterator of neighbors of v
	virtual const ::std::vector<int>::const_iterator endNeighs(int v) const = 0;

	void printGraphInfo() const;
	void printBenchmarkInfo() const;
};

#endif // !_GRAPH_REPRESENTATION_H
