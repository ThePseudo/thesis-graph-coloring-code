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
	size_t rows;
	size_t cols;

	std::vector<size_t> col_idxs;
	size_t* row_ptrs;

	void init(size_t n_rows, size_t n_cols);

	template<typename Iterator>
	void populateRow(size_t row_idx, Iterator begin, const Iterator end);

public:
	CompressedSparseRow();
	CompressedSparseRow(const CompressedSparseRow& to_copy);
	CompressedSparseRow(size_t rows, size_t cols);
	CompressedSparseRow(size_t n);
	~CompressedSparseRow();

	friend std::istream& operator>>(std::istream& is, CompressedSparseRow& m);

	const size_t nE() const override;
	const size_t nV() const override;
	const bool get(size_t row, size_t col) const override;

	const int countNeighs(size_t v) const { __super::countNeighs(v); };

	const ::std::vector<size_t>::const_iterator beginNeighs(size_t row_idx) const override;
	const ::std::vector<size_t>::const_iterator endNeighs(size_t row_idx) const override;
};

#endif // !_COMPRESSED_SPARSE_ROW_H