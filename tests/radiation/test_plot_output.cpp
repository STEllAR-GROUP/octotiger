// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
#include "octotiger/test_problems/radiation/plot_output.hpp"
#include <chrono>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
std::vector<std::string> row(std::istream& stream) {
    std::string line, value;
    std::getline(stream, line);
    std::istringstream values(line);
    std::vector<std::string> result;
    while (std::getline(values, value, ',')) {
        result.push_back(value);
    }
    return result;
}
struct AbsentReference {
    double operator[](size_t) const {
        throw std::runtime_error("Numerical-only export read a missing reference");
    }
};
}

int main() {
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() / ("octotiger-slice-export-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(path / "radiation-slices");
        fs::create_directories(path / "marshak-material-slices");
        for (bool reference : {false, true}) {
            radiationTests::SliceOutput radiation(true, path.string(), 3., 1., 2., 4, reference);
            auto read = [](int f) { return 10. + f; };
            if (reference) {
                const std::array<double, 6> values{100, 101, 20, 21, 22, 23};
                radiation.capture(.5, .5, .5, values, 2, read);
            } else {
                radiation.capture(.5, .5, .5, AbsentReference{}, 2, read);
            }
            radiation.finish();
            std::ifstream file(path / "radiation-slices/slice-level-4-1-1.csv");
            const auto header = row(file), data = row(file);
            require(header.size() == (reference ? 13 : 9), "Wrong radiation column count");
            require(data.size() == header.size(), "Wrong radiation data width");
            require(header[5] == "er" && std::stod(data[5]) == 10, "Numerical energy changed");
            if (reference) {
                require(header[9] == "er_ref" && std::stod(data[9]) == 20, "Wrong reference offset");
                require(header[12] == "fz_ref" && std::stod(data[12]) == 23, "Wrong reference flux");
            }
            radiationTests::MaterialSliceOutput material(true, path.string(), 3., 1., 2., 4, reference);
            material.capture(.5, .5, .5, 9, 8);
            material.finish();
            std::ifstream gas(path / "marshak-material-slices/slice-level-4-1-1.csv");
            const auto gas_header = row(gas), gas_data = row(gas);
            require(gas_header.size() == (reference ? 7 : 6), "Wrong material column count");
            require(gas_data.size() == gas_header.size() && std::stod(gas_data[5]) == 9, "Material energy changed");
            if (reference) {
                require(gas_header[6] == "egas_ref" && std::stod(gas_data[6]) == 8, "Missing material reference");
            }
        }
        fs::remove_all(path);
        radiationTests::SliceOutput disabled(true, path.string(), 3., 1., 2.);
        disabled.capture(.5, .5, .5, AbsentReference{}, 0,
            [](int) -> double { throw std::runtime_error("Disabled output read numerical data"); });
        require(!fs::exists(path), "Disabled export created output");
        std::cout << "Reference and numerical-only slice export passed\n";
    } catch (const std::exception& error) {
        fs::remove_all(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
