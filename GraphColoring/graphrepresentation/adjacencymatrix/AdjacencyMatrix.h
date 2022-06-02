#ifndef _ADJACENCY_MATRIC_H
#define _ADJACENCY_MATRIC_H

#ifdef GRAPH_REPRESENTATION_ADJ_MATRIX

#include "GraphRepresentation.h"

#include <vector>

#if defined(PARALLEL_INPUT_LOAD) || defined(PARTITIONED_INPUT_LOAD)
#include <mutex>
#endif


class AdjacencyMatrix : public GraphRepresentation {
private:
	std::vector<std::vector<int>> adj;

	bool parseInput(std::istream&);
#ifdef PARALLEL_INPUT_LOAD
	void parseInputParallel(std::istream&, std::mutex&);
#endif
#ifdef PARTITIONED_INPUT_LOAD
	void parseInputParallel(std::string&, std::mutex&, int const, size_t const);
#endif
	bool readHeader(std::istream&, struct header&);
#ifdef PARALLEL_INPUT_LOAD
	bool readVertex(std::istream&, struct vertex&, std::mutex&);
#endif
#ifdef PARTITIONED_INPUT_LOAD
	bool readVertex(std::string&, struct vertex&);
#endif
#ifdef SEQUENTIAL_INPUT_LOAD
	bool readVertex(std::istream&, struct vertex&);
#endif

public:
	AdjacencyMatrix();
	AdjacencyMatrix(const AdjacencyMatrix& to_copy);
	AdjacencyMatrix(int nV);
	~AdjacencyMatrix();

	friend std::istream& operator>>(std::istream& is, AdjacencyMatrix& m);

	bool get(int v, int w) const override;

	const ::std::vector<int>::const_iterator beginNeighs(int v) const override;
	const ::std::vector<int>::const_iterator endNeighs(int v) const override;
};

#endif

#endif // !_ADJACENCY_MATRIC_H