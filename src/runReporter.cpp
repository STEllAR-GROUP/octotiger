//  Copyright (c) 2026 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0.

#include "octotiger/runReporter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace octotiger {
namespace {

struct AnalyticReport {
	Real l1;
	Real l2;
	Real linf;
};

struct UnitReport {
	Real grams;
	Real centimeters;
	Real seconds;
	Real lightSpeed;
};

struct ReporterState {
	std::mutex mutex;
	std::string problemName;
	std::string detailedLogPath;
	std::string resultsPath;
	int refinementFrequency = 1;
	bool initialized = false;
	Real finalTime = 0.0;
	std::optional<StepReport> lastStep;
	std::optional<IntervalReport> lastInterval;
	std::optional<UnitReport> units;
	std::map<std::string, AnalyticReport> analytic;
};

ReporterState& state() {
	static ReporterState reporterState;
	return reporterState;
}

std::string escapedJson(const std::string& value) {
	std::ostringstream result;
	for (const char character : value) {
		switch (character) {
		case '\\': result << "\\\\"; break;
		case '"': result << "\\\""; break;
		case '\n': result << "\\n"; break;
		case '\r': result << "\\r"; break;
		case '\t': result << "\\t"; break;
		default: result << character;
		}
	}
	return result.str();
}

void writeResults(const ReporterState& reporterState, const char* status) {
	if (reporterState.resultsPath.empty()) {
		return;
	}
	const std::filesystem::path target(reporterState.resultsPath);
	const auto parent = target.parent_path();
	if (!parent.empty()) {
		std::filesystem::create_directories(parent);
	}
	const auto temporary = target.string() + ".tmp";
	std::ofstream output(temporary, std::ios::out | std::ios::trunc);
	if (!output) {
		throw std::runtime_error("Cannot write structured run results to " + temporary);
	}
	output << std::setprecision(17);
	output << "{\n  \"schemaVersion\": 1,\n";
	output << "  \"status\": \"" << status << "\",\n";
	output << "  \"problem\": \"" << escapedJson(reporterState.problemName) << "\",\n";
	output << "  \"finalTime\": " << double(reporterState.finalTime) << ",\n";
	output << "  \"lastStep\": ";
	if (reporterState.lastStep) {
		const auto& step = *reporterState.lastStep;
		output << "{\"number\": " << step.step << ", \"time\": " << double(step.time)
			<< ", \"dt\": " << double(step.timeStep) << ", \"wallSeconds\": "
			<< step.wallSeconds << ", \"rotationalTime\": " << double(step.rotationalTime) << "}";
	} else {
		output << "null";
	}
	output << ",\n  \"limiter\": ";
	if (reporterState.lastInterval) {
		const auto& interval = *reporterState.lastInterval;
		output << "{\"x\": " << double(interval.x) << ", \"y\": " << double(interval.y)
			<< ", \"z\": " << double(interval.z) << ", \"a\": " << double(interval.speed)
			<< ", \"ur\": " << double(interval.rightDensity) << ", \"ul\": "
			<< double(interval.leftDensity) << ", \"vr\": " << double(interval.rightVelocity)
			<< ", \"vl\": " << double(interval.leftVelocity) << ", \"dimension\": "
			<< interval.dimension << "}";
	} else {
		output << "null";
	}
	output << ",\n  \"mesh\": ";
	if (reporterState.lastInterval) {
		const auto& interval = *reporterState.lastInterval;
		output << "{\"grids\": " << interval.grids << ", \"leaves\": " << interval.leaves
			<< ", \"amrBoundaries\": " << interval.amrBoundaries << "}";
	} else {
		output << "null";
	}
	output << ",\n  \"analytic\": {";
	bool first = true;
	for (const auto& [field, analytic] : reporterState.analytic) {
		output << (first ? "\n" : ",\n") << "    \"" << escapedJson(field) << "\": {\"l1\": "
			<< double(analytic.l1) << ", \"l2\": " << double(analytic.l2)
			<< ", \"linf\": " << double(analytic.linf) << "}";
		first = false;
	}
	if (!first) {
		output << '\n';
	}
	output << "  },\n  \"units\": ";
	if (reporterState.units) {
		const auto& units = *reporterState.units;
		output << "{\"grams\": " << double(units.grams) << ", \"centimeters\": "
			<< double(units.centimeters) << ", \"seconds\": " << double(units.seconds)
			<< ", \"lightSpeed\": " << double(units.lightSpeed) << "}";
	} else {
		output << "null";
	}
	output << "\n}\n";
	output.close();
	std::error_code removeError;
	std::filesystem::remove(target, removeError);
	std::filesystem::rename(temporary, target);
}

void appendDetailed(const ReporterState& reporterState, const std::string& text) {
	if (reporterState.detailedLogPath.empty()) {
		return;
	}
	std::ofstream output(reporterState.detailedLogPath, std::ios::out | std::ios::app);
	if (!output) {
		throw std::runtime_error("Cannot write detailed run log to " + reporterState.detailedLogPath);
	}
	output << text << '\n';
}

} // namespace

RunReporter& RunReporter::instance() {
	static RunReporter reporter;
	return reporter;
}

void RunReporter::initialize(const std::string& problemName, const std::string& detailedLogPath,
	const std::string& resultsPath, int refinementFrequency, Real initialTime,
	Real initialRotationalTime) {
	auto& reporterState = state();
	std::lock_guard<std::mutex> lock(reporterState.mutex);
	reporterState.problemName = problemName;
	reporterState.detailedLogPath = detailedLogPath;
	reporterState.resultsPath = resultsPath;
	reporterState.refinementFrequency = std::max(1, refinementFrequency);
	reporterState.finalTime = initialTime;
	reporterState.lastStep.reset();
	reporterState.lastInterval.reset();
	reporterState.units.reset();
	reporterState.analytic.clear();
	reporterState.initialized = true;
	if (!detailedLogPath.empty()) {
		const std::filesystem::path logPath(detailedLogPath);
		if (!logPath.parent_path().empty()) {
			std::filesystem::create_directories(logPath.parent_path());
		}
		std::ofstream(logPath, std::ios::out | std::ios::trunc);
	}
	const bool isBinary = problemName == "DWD";
	const std::string displayName = isBinary
		? "Binary evolution [legacy problem ID: DWD]" : problemName;
	std::ostringstream console;
	console << "\nOcto-Tiger | " << displayName << '\n';
	console << std::setw(4) << "step" << "  " << std::setw(12) << "time [code]" << "  "
		<< std::setw(12) << "Δt [code]" << "  " << std::setw(10) << "wall [s]" << "  "
		<< std::setw(16) << (isBinary ? "time [P₀]" : "rotation [turns]") << '\n';
	console << std::setw(4) << 0 << "  " << std::scientific << std::setprecision(6)
		<< std::setw(12) << double(initialTime) << "           -            -  "
		<< std::fixed << std::setprecision(6) << std::setw(16)
		<< double(initialRotationalTime / (2.0 * std::acos(-1.0))) << '\n';
	std::cout << console.str();
	appendDetailed(reporterState, "run started: problem=" + problemName);
	writeResults(reporterState, "running");
}

void RunReporter::reportStep(const StepReport& report) {
	auto& reporterState = state();
	std::lock_guard<std::mutex> lock(reporterState.mutex);
	if (!reporterState.initialized) return;
	std::ostringstream console;
	if ((report.step - 1) % reporterState.refinementFrequency == 0) {
		const int firstStep = report.step;
		const int lastStep = firstStep + reporterState.refinementFrequency - 1;
		console << "\nrefinement interval " << firstStep << '-' << lastStep << '\n';
	}
	console << std::setw(4) << report.step << "  " << std::scientific << std::setprecision(6)
		<< std::setw(12) << double(report.time) << "  " << std::setw(12) << double(report.timeStep)
		<< "  " << std::fixed << std::setprecision(4) << std::setw(10) << report.wallSeconds
		<< "  " << std::setprecision(6) << std::setw(16)
		<< double(report.rotationalTime / (2.0 * std::acos(-1.0))) << '\n';
	std::cout << console.str();
	reporterState.lastStep = report;
	reporterState.finalTime = report.time;
	std::ostringstream detail;
	detail << std::setprecision(17) << "step " << report.step << ": time=" << double(report.time)
		<< " dt=" << double(report.timeStep) << " wallSeconds=" << report.wallSeconds
		<< " rotationalTime=" << double(report.rotationalTime);
	appendDetailed(reporterState, detail.str());
	writeResults(reporterState, "running");
}

void RunReporter::reportInterval(const IntervalReport& report) {
	auto& reporterState = state();
	std::lock_guard<std::mutex> lock(reporterState.mutex);
	if (!reporterState.initialized) return;
	reporterState.lastInterval = report;
	std::ostringstream console;
	console << "interval summary | wall " << std::fixed << std::setprecision(3) << report.wallSeconds
		<< " s | grids " << report.grids << " (" << report.leaves << " leaves) | AMR boundaries "
		<< report.amrBoundaries << '\n';
	std::cout << console.str();
	std::ostringstream detail;
	detail << std::setprecision(17) << "interval ending at step " << report.lastStep
		<< ": limiter=(" << double(report.x) << ',' << double(report.y) << ',' << double(report.z)
		<< ") a=" << double(report.speed) << " ur=" << double(report.rightDensity)
		<< " ul=" << double(report.leftDensity) << " vr=" << double(report.rightVelocity)
		<< " vl=" << double(report.leftVelocity) << " dimension=" << report.dimension
		<< " grids=" << report.grids << " leaves=" << report.leaves
		<< " amrBoundaries=" << report.amrBoundaries;
	appendDetailed(reporterState, detail.str());
	writeResults(reporterState, "running");
}

void RunReporter::reportAction(const std::string& action, const std::string& detail) {
	auto& reporterState = state();
	std::lock_guard<std::mutex> lock(reporterState.mutex);
	if (!reporterState.initialized) return;
	appendDetailed(reporterState, "action: " + action + (detail.empty() ? "" : " | " + detail));
}

void RunReporter::reportUnits(Real grams, Real centimeters, Real seconds, Real lightSpeed) {
	auto& reporterState = state();
	std::lock_guard<std::mutex> lock(reporterState.mutex);
	if (!reporterState.initialized) return;
	reporterState.units = UnitReport {grams, centimeters, seconds, lightSpeed};
	writeResults(reporterState, "running");
}

void RunReporter::reportAnalytic(const std::string& field, Real l1, Real l2, Real linf) {
	auto& reporterState = state();
	std::lock_guard<std::mutex> lock(reporterState.mutex);
	if (!reporterState.initialized) return;
	reporterState.analytic[field] = {l1, l2, linf};
	std::ostringstream detail;
	detail << std::setprecision(17) << "analytic " << field << ": l1=" << double(l1)
		<< " l2=" << double(l2) << " linf=" << double(linf);
	appendDetailed(reporterState, detail.str());
	writeResults(reporterState, "running");
}

void RunReporter::finish(Real finalTime) {
	auto& reporterState = state();
	std::lock_guard<std::mutex> lock(reporterState.mutex);
	if (!reporterState.initialized) return;
	reporterState.finalTime = finalTime;
	appendDetailed(reporterState, "run completed");
	writeResults(reporterState, "completed");
	std::ostringstream console;
	console << "\ncompleted | t=" << std::scientific << std::setprecision(6)
		<< double(finalTime) << '\n';
	std::cout << console.str();
}

} // namespace octotiger
