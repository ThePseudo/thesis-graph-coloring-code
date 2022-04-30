#ifndef _CUSPARSE_COLORING_
#define _CUSPARSE_COLORING_

#include "configuration.h"
#include "ColoringAlgorithm.h"

class CusparseColoring : public ColoringAlgorithm {
private:

public:
	CusparseColoring(std::string const filepath);
	const int startColoring() override;
};

#endif // !_CUSPARSE_COLORING_

