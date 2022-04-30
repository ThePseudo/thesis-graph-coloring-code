#ifndef _COMPRESSED_SPARSE_ROW_H
#define _COMPRESSED_SPARSE_ROW_H

#include "GraphRepresentation.h"

#include <istream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "configuration.h"

class CompressedSparseRow : public GraphRepresentation {
private:
	int rows;
	int cols;

	std::vector<int> col_idxs;
	int* row_ptrs;

	void init(int n_rows, int n_cols);

	template<typename Iterator>
	void populateRow(int row_idx, Iterator begin, const Iterator end);

public:
	CompressedSparseRow();
	CompressedSparseRow(const CompressedSparseRow& to_copy);
	CompressedSparseRow(int rows, int cols);
	CompressedSparseRow(int n);
	~CompressedSparseRow();

	const int* getColIndexes() const;
	const int* getRowPointers() const;

	friend std::istream& operator>>(std::istream& is, CompressedSparseRow& m);

	const size_t nE() const override;
	const int nV() const override;
	const bool get(int row, int col) const override;

	const ::std::vector<int>::const_iterator beginNeighs(int row_idx) const override;
	const ::std::vector<int>::const_iterator endNeighs(int row_idx) const override;
};

#endif // !_COMPRESSED_SPARSE_ROW_H