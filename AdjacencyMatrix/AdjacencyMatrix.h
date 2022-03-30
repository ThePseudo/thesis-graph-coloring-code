#ifndef _ADJACENCY_MATRIC_H
#define _ADJACENCY_MATRIC_H

#include "GraphRepresentation.h"

#include <vector>

#include "configuration.h"

#if defined(PARALLEL_INPUT_LOAD) || defined(PARTITIONED_INPUT_LOAD)
#include <mutex>
#endif


class AdjacencyMatrix :	public GraphRepresentation {
private:
	std::vector<std::vector<size_t>> adj;
	size_t _nV, _nE;

	bool parseInput(std::istream&);
#ifdef PARALLEL_INPUT_LOAD
	void parseInputParallel(std::istream&, std::mutex&);
#endif
#ifdef PARTITIONED_INPUT_LOAD
	void parseInputParallel(std::string& const, std::mutex&, int const, size_t const);
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
	AdjacencyMatrix(size_t nV);
	~AdjacencyMatrix();

	friend std::istream& operator>>(std::istream& is, AdjacencyMatrix& m);

	const size_t nE() const override;
	const size_t nV() const override;
	const bool get(size_t v, size_t w) const throw(std::out_of_range) override;

	const int countNeighs(size_t v) const throw(std::out_of_range) { __super::countNeighs(v); };

	const ::std::vector<size_t>::const_iterator beginNeighs(int v) const throw(std::out_of_range) override;
	const ::std::vector<size_t>::const_iterator endNeighs(int v) const throw(std::out_of_range) override;
};

#endif // !_ADJACENCY_MATRIC_H