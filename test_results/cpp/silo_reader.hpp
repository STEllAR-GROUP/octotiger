#pragma once
#include "common.hpp"
namespace rr
{
struct Cell {
	std::array<double, 3> lo, hi;
	double value;
};
struct Snapshot {
	double time = 0;
	std::vector<Cell> cells;
};
Snapshot read_silo(const fs::path &, const Options &, const Json &);
} // namespace rr
