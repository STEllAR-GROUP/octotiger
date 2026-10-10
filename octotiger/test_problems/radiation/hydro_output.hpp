// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
#pragma once
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <string>

namespace radiationHydroTests {
// Numerical output only: these coupled problems have no supplied exact solution.
// The runner opts in by creating this directory. Each leaf owns a separate file.
class SliceOutput {
    std::filesystem::path directory_;
    std::ofstream stream_;
    double time_, dx_, length_;
    bool enabled_;
    int level_;
public:
    SliceOutput(const std::string& dataDir, double time, double dx, double length, int level = -1) :
        directory_(std::filesystem::path(dataDir) / "radiation-hydro-slices"),
        time_(time), dx_(dx), length_(length), enabled_(std::filesystem::is_directory(directory_)), level_(level) {}

    void capture(double x, double y, double z, const std::array<double, 10>& fields) {
        if (!enabled_ || std::abs(z - dx_ / 2) > 128 * std::numeric_limits<double>::epsilon() * length_) {
            return;
        }
        if (!stream_.is_open()) {
            const auto i = std::llround((x + length_ / 2) / dx_ - .5);
            const auto j = std::llround((y + length_ / 2) / dx_ - .5);
            stream_.exceptions(std::ios::badbit | std::ios::failbit);
            stream_.imbue(std::locale::classic());
            const auto prefix = level_ < 0 ? std::string("slice-") : "slice-level-" + std::to_string(level_) + "-";
            stream_.open(directory_ / (prefix + std::to_string(i) + "-" + std::to_string(j) + ".csv"));
            stream_ << "t,dx,x,y,z,rho,vx,vy,vz,gas_temperature,radiation_temperature,er,fx,fy,fz\n"
                << std::scientific << std::setprecision(17);
        }
        stream_ << time_ << ',' << dx_ << ',' << x << ',' << y << ',' << z;
        for (double value : fields) {
            stream_ << ',' << value;
        }
        stream_ << '\n';
    }
    void finish() {
        if (stream_.is_open()) {
            stream_.close();
        }
    }
};
}
