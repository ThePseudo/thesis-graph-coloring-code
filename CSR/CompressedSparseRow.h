#ifndef _COMPRESSED_SPARSE_ROW_H
#define _COMPRESSED_SPARSE_ROW_H

#include <istream>
#include <iterator>
#include <stdexcept>
#include <vector>

class CompressedSparseRow {
private:
	size_t rows;
	size_t cols;

	std::vector<size_t> col_idxs;
	size_t* row_ptrs;

	void init(size_t n_rows, size_t n_cols);

	template<typename Iterator>
	void populateRow(size_t row_idx, Iterator begin, const Iterator end) throw(std::invalid_argument, std::out_of_range);

public:
	CompressedSparseRow();
	CompressedSparseRow(const CompressedSparseRow& to_copy);
	CompressedSparseRow(size_t rows, size_t cols);
	CompressedSparseRow(size_t n);
	~CompressedSparseRow();

	friend std::istream& operator>>(std::istream& is, CompressedSparseRow& m);

	const size_t count() const;
	const size_t countRows() const;
	const bool get(size_t row, size_t col) const throw(std::out_of_range);

	const ::std::vector<size_t>::const_iterator beginRow(int row_idx) const throw(std::out_of_range);
	const ::std::vector<size_t>::const_iterator endRow(int row_idx) const throw(std::out_of_range);
};

#endif // !_COMPRESSED_SPARSE_ROW_H