//  Copyright (c) 2026 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0.

#ifndef OCTOTIGER_RUN_REPORTER_HPP_
#define OCTOTIGER_RUN_REPORTER_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/math/Real.hpp"

#include <string>

namespace octotiger {

struct StepReport {
	int step = 0;
	Real time = 0.0;
	Real timeStep = 0.0;
	double wallSeconds = 0.0;
	Real rotationalTime = 0.0;
};

struct IntervalReport {
	int lastStep = 0;
	double wallSeconds = 0.0;
	Real x = 0.0;
	Real y = 0.0;
	Real z = 0.0;
	Real speed = 0.0;
	Real rightDensity = 0.0;
	Real leftDensity = 0.0;
	Real rightVelocity = 0.0;
	Real leftVelocity = 0.0;
	int dimension = 0;
	int grids = 0;
	int leaves = 0;
	int amrBoundaries = 0;
};

class OCTOTIGER_EXPORT RunReporter {
public:
	static RunReporter& instance();

	void initialize(const std::string& problemName, const std::string& detailedLogPath,
		const std::string& resultsPath, int refinementFrequency, Real initialTime,
		Real initialRotationalTime);
	void reportStep(const StepReport& report);
	void reportInterval(const IntervalReport& report);
	void reportAction(const std::string& action, const std::string& detail = {});
	void reportUnits(Real grams, Real centimeters, Real seconds, Real lightSpeed);
	void reportAnalytic(const std::string& field, Real l1, Real l2, Real linf);
	void finish(Real finalTime);

private:
	RunReporter() = default;
};

} // namespace octotiger

#endif
