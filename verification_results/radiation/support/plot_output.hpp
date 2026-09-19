// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
#pragma once
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
	std::filesystem::path directory_;
	std::ofstream stream_;
	double time_, dx_, length_;
	bool enabled_;
public:
	SliceOutput(bool regression, std::string const& dataDir, double time,
			double dx, double length) :
		directory_(std::filesystem::path(dataDir) / "radiation-slices"),
		time_(time), dx_(dx), length_(length),
		enabled_(regression && std::filesystem::is_directory(directory_)) {}
	template<class Reference, class ReadField>
	void capture(double x, double y, double z, Reference const& reference,
			int offset, ReadField read) {
		if (!enabled_ || std::abs(z - dx_ / 2) >
				128 * std::numeric_limits<double>::epsilon() * length_) return;
		if (!stream_.is_open()) {
			auto const i = std::llround((x + length_ / 2) / dx_ - .5);
			auto const j = std::llround((y + length_ / 2) / dx_ - .5);
			stream_.exceptions(std::ios::badbit | std::ios::failbit);
			stream_.imbue(std::locale::classic());
			stream_.open(directory_ / ("slice-" + std::to_string(i) + "-" +
				std::to_string(j) + ".csv"), std::ios::out | std::ios::trunc);
			stream_ << "t,dx,x,y,z,er,fx,fy,fz,er_ref,fx_ref,fy_ref,fz_ref\n"
				<< std::scientific << std::setprecision(17);
		}
		stream_ << time_ << ',' << dx_ << ',' << x << ',' << y << ',' << z;
		for (int f = 0; f < 4; ++f) stream_ << ',' << read(f);
		for (int f = 0; f < 4; ++f) stream_ << ',' << double(reference[offset + f]);
		stream_ << '\n';
	}
	void finish() { if (stream_.is_open()) stream_.close(); }
};
}
