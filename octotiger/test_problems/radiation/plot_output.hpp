// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
#pragma once
#include "octotiger/math/Debug.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>

namespace radiationTests {
// Optional final-comparison export. The runner creates this directory to opt in.
// One file per leaf intersecting z=dx/2; no shared stream or HPX reductions.
// Numerical values are read before compute_analytic replaces them with reference.
class SliceOutput {
	std::filesystem::path directoryPath;
	std::ofstream outputStream;
	double timeValue, dx, length;
	bool enabled;
public:
	SliceOutput(bool regression, std::string const& dataDir, double time,
			double dx, double length) :
		directoryPath(std::filesystem::path(dataDir) / "radiation-slices"),
		timeValue(time), dx(dx), length(length),
		enabled(regression && std::filesystem::is_directory(directoryPath)) {}
	template<class Reference, class ReadField>
	void capture(double x, double y, double z, Reference const& reference,
			int offset, ReadField read) {
		FpeGuard fpeGuard{};
		if (!enabled || std::abs(z - dx / 2) >
				128 * std::numeric_limits<double>::epsilon() * length) return;
		if (!outputStream.is_open()) {
			auto const i = std::llround((x + length / 2) / dx - .5);
			auto const j = std::llround((y + length / 2) / dx - .5);
			outputStream.exceptions(std::ios::badbit | std::ios::failbit);
			outputStream.imbue(std::locale::classic());
			outputStream.open(directoryPath / ("slice-" + std::to_string(i) + "-" +
				std::to_string(j) + ".csv"), std::ios::out | std::ios::trunc);
			outputStream << "t,dx,x,y,z,er,fx,fy,fz,er_ref,fx_ref,fy_ref,fz_ref\n"
				<< std::scientific << std::setprecision(17);
		}
		outputStream << timeValue << ',' << dx << ',' << x << ',' << y << ',' << z;
		for (int f = 0; f < 4; ++f) outputStream << ',' << read(f);
		for (int f = 0; f < 4; ++f) outputStream << ',' << double(reference[offset + f]);
		outputStream << '\n';
	}
	void finish() { if (outputStream.is_open()) outputStream.close(); }
};
}
