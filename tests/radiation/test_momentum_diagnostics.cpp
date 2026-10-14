// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
#include "octotiger/radiation/momentum_diagnostics.hpp"
#include <iostream>
#include <stddef.h>
#include <sstream>
#include <string>
#include <vector>

namespace {
size_t checks = 0;
void require(bool value, const std::string& message) {
    ++checks;
    if (!value) {
        throw std::runtime_error(message);
    }
}
std::vector<std::string> row(const radiationConservation::Totals& values,
    const radiationMomentum::Parameters& parameters) {
    std::ostringstream out;
    radiationMomentum::writeCsvRow(out, 2, values, parameters);
    std::istringstream input(out.str());
    std::vector<std::string> columns;
    std::string value;
    while (std::getline(input, value, ',')) {
        columns.push_back(value);
    }
    require(columns.size() == 25, "Momentum ledger schema has 25 columns");
    return columns;
}
void near(const std::string& value, double expected) {
    require(std::abs(std::stod(value) - expected) < 1.e-14 * std::max(1., std::abs(expected)),
        "Unit-converted momentum ledger value");
}
}

int main() {
    try {
        radiationConservation::Totals values;
        values.volume = 2;
        values.gas = {{3, 2, 0, 0}};
        values.value = {{7, 50, 0, 0}};
        values.gas_boundary = {{0, 1, 0, 0}};
        values.boundary = {{0, 10, 0, 0}};
        values.source = {{0, -20, 0, 0}};
        radiationMomentum::Parameters parameters{5, 2, 3, 4, false, false, false, false};
        auto result = row(values, parameters);
        near(result[0], 8);
        near(result[1], 3.75);
        near(result[2], 54);
        near(result[3], 3.375);
        near(result[4], 7.875);
        near(result[5], 3);
        near(result[8], 3);
        near(result[11], 1.5);
        require(result[14] == "1", "M1 radiation boundary is available");
        near(result[15], .6);
        require(result[18] == "1", "Coupled internal exchange needs no external correction");
        near(result[19], 0);
        parameters.fixed_material = true;
        result = row(values, parameters);
        near(result[19], -1.2);
        parameters.fld = true;
        result = row(values, parameters);
        require(result[14] == "0" && result[15].empty() && result[16].empty() && result[17].empty(),
            "FLD unrecorded traction is unavailable, not zero");
        require(result[18] == "0" && result[19].empty() && result[20].empty() && result[21].empty(),
            "FLD unrecorded fixed-matter impulse is unavailable");
        parameters.periodic = true;
        result = row(values, parameters);
        require(result[14] == "1", "Periodic FLD has exactly no physical boundary");
        near(result[15], 0);
        parameters.normalized = true;
        result = row(values, parameters);
        near(result[0], 2);
        near(result[1], 5);
        near(result[3], 3);
        near(result[8], 2);
        require(result[24] == "normalized\n", "Normalized units explicitly named");
        radiationConservation::Ledger ledger;
        values.gas_boundary[1] = 7;
        ledger.consume(values);
        values.gas_boundary[1] = 11;
        const auto sampled = ledger.consume(values);
        require(sampled.gas_boundary[1] == 11, "Cumulative gas boundaries are not accumulated twice");
        require(sampled.boundary[1] == 20, "Radiation interval boundaries still accumulate");
        auto combined = values;
        combined += values;
        require(combined.gas[1] == 4 && combined.gas_boundary[1] == 22,
            "Leaf reductions include physical gas momentum and boundaries");
        std::cout << "Momentum diagnostic tests passed (" << checks << " checks)\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
