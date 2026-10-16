#include <cerrno>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>

#include "common.hpp"

// Standalone reporting repair for the diagnosed L2 FLD blast marker mismatch.
// Link this main against a separately built rr_core containing the corrected
// read_fld_diagnostics. It never executes the solver or writes into the run.
namespace {
namespace fs = std::filesystem;
using rr::Json;

constexpr double stop_time = 22000.;
constexpr double hard_dt = 110.;
std::string const executable_hash =
    "ca38f12e338eed0b7074dbb15c70c089a8d31d1c12aa87f5b044c42824721f0b";
std::string const diagnosed_error = "Wrong FLD completion marker/time";

void regular_file(fs::path const &path)
{
    rr::require(fs::is_regular_file(path) && !fs::is_symlink(path) && fs::file_size(path) > 0,
        "Expected a nonempty regular file without a symlink: " + path.string());
}

bool descendant(fs::path const &path, fs::path const &directory)
{
    auto part = path.begin();
    for (auto const &component : directory) {
        if (part == path.end() || *part != component) {
            return false;
        }
        ++part;
    }
    return true;
}

Json expected_config(fs::path const &leaf)
{
    return {{"atomic_mass", "1"}, {"atomic_number", "1"}, {"cfl", "0.4"},
        {"code_to_cm", "1"}, {"code_to_g", "1"}, {"code_to_s", "1"},
        {"correct_am_hydro", "off"}, {"datadir", leaf.string() + "/"},
        {"disable_analytic", "on"}, {"disable_diagnostics", "on"}, {"disable_output", "off"},
        {"eos", "IDEAL"}, {"gravity", "off"}, {"hard_dt", "110"}, {"hydro", "on"},
        {"hydro_device_kernel_type", "OFF"}, {"hydro_host_kernel_type", "LEGACY"},
        {"inflow_bc", "off"}, {"max_level", "2"}, {"min_level", "2"},
        {"monopole_device_kernel_type", "OFF"}, {"multipole_device_kernel_type", "OFF"},
        {"n_species", "1"}, {"odt", "183.33333333333334"}, {"omega", "0"},
        {"periodic", "off"}, {"problem", "RADIATION_BLAST_WAVE"}, {"rad_implicit", "on"},
        {"radiation", "on"}, {"radiation_backend", "cpu"},
        {"radiation_integrator", "backward_euler"}, {"radiation_limiter", "levermore_pomraning"},
        {"radiation_model", "fld"}, {"radiation_reduced_c_factor", "1"}, {"reflect_bc", "off"},
        {"stop_time", "22000"}, {"unigrid", "on"}, {"xscale", "10000000000000"}};
}

void validate_identity(fs::path const &leaf, Json const &metadata)
{
    Json const expected{{"status", "failed"}, {"error", diagnosed_error},
        {"case", "radiating_blast"}, {"level", 2}, {"inx", 8}, {"cells", 32},
        {"length", 2.e13}, {"dx", 6.25e11}, {"time", stop_time}, {"benchmark_time", stop_time},
        {"background", 0.}, {"c", rr::c_cgs}, {"physical_c", rr::c_cgs},
        {"transport_c", rr::c_cgs}, {"units", rr::cgs_units},
        {"origin", "Octo-TIGER application"}, {"radiation_model", "fld"},
        {"radiation_backend", "cpu"}, {"radiation_integrator", "backward_euler"},
        {"radiation_algorithm_version", 2}, {"radiation_limiter", "levermore_pomraning"},
        {"radiation_reduced_c_factor", 1.}, {"radiation_frame", "mixed"},
        {"radiation_hydro_scheme", "first_order_conservative"},
        {"radiation_equations", "mixed-frame physical-c FLD radiation hydrodynamics"},
        {"radiation_flux_semantics", "derived_fld_lab_flux"},
        {"reference_kind", "unavailable"}, {"reference_model", "none"},
        {"reference_sampling", "none"}, {"reference_exact_for_solver", false},
        {"executable_sha256", executable_hash},
        {"movie_capture", {{"hard_dt", hard_dt}, {"odt", stop_time / 120.},
                              {"requested_snapshots", 121}, {"steps_per_output_check", 1}}}};
    rr::require(metadata.is_object(), "Run metadata is not an object");
    for (auto const &[key, value] : expected.items()) {
        rr::require(metadata.contains(key) && metadata.at(key) == value,
            "Diagnosed run identity differs: " + key);
    }
    rr::require(!metadata.value("adaptive", false) && !metadata.contains("checkpoint_resume") &&
            !fs::exists(leaf / "restart-provenance.json") && !metadata.contains("norms") &&
            !metadata.contains("fld_diagnostics") && !metadata.contains("conservation"),
        "Expected the original full-run failure before derived reporting metadata");

    auto const config = rr::read_config(leaf / "run.ini");
    rr::require(config == expected_config(leaf) && metadata.at("config") == config,
        "Original configuration differs from the diagnosed physical-c FLD blast");
    auto signature = config;
    signature["radiation_algorithm_version"] = 2;
    for (auto const *key : {"max_level", "min_level", "datadir", "rad_reference", "disable_output",
             "odt", "radiation_backend"}) {
        signature.erase(key);
    }
    rr::require(metadata.at("comparison_signature") == signature,
        "Saved comparison signature does not describe the original configuration");

    auto const batch = leaf.parent_path().parent_path();
    auto const campaign = batch.parent_path().parent_path();
    auto const remote_root = campaign.parent_path();
    rr::require(leaf.filename() == "l2" && leaf.parent_path().filename() == "radiating_blast" &&
            batch.filename() == "l2-radiating_blast-fld" && batch.parent_path().filename() == "runs" &&
            campaign.filename() == "campaign",
        "Unexpected original campaign run path");
    auto const launcher = campaign / "control/launch-solver";
    auto const executable = remote_root / "build/octotiger";
    Json const expected_command = Json::array({launcher.string(), executable.string(),
        "--config_file=" + (leaf / "run.ini").string(), "--hpx:threads=64"});
    rr::require(metadata.at("command") == expected_command, "Recorded solver invocation differs");
    regular_file(launcher);
    regular_file(executable);
    rr::require(rr::sha256(executable) == executable_hash,
        "Original solver executable no longer matches its recorded provenance");
}

void validate_marker(fs::path const &leaf)
{
    std::istringstream input(rr::read_text(leaf / "run.log"));
    std::string line;
    std::smatch match;
    std::regex const marker(
        R"(^RADIATION_FLD_FINISHED RADIATION_BLAST_WAVE t=(\S+) requested=(\S+)\s*$)");
    size_t found = 0;
    while (std::getline(input, line)) {
        if (line.starts_with("RADIATION_FLD_FINISHED")) {
            rr::require(std::regex_match(line, match, marker) &&
                    rr::numeric(match[1]) == stop_time && rr::numeric(match[2]) == stop_time,
                "Original production completion marker or requested time differs");
            ++found;
        }
    }
    rr::require(found == 1, "Expected exactly one original production completion marker");
}

void validate_snapshots(fs::path const &leaf)
{
    auto const snapshots = rr::numerical_silos(leaf);
    rr::require(snapshots.size() == 121, "Expected all 121 original numerical states");
    for (size_t i = 0; i < snapshots.size(); ++i) {
        auto const &path = snapshots[i];
        rr::require(path.filename() == (i == 120 ? "final.silo" : "X." + std::to_string(i) + ".silo"),
            "Original numerical snapshots are not contiguous");
        regular_file(path);
        auto const shards = fs::path(path.string() + ".data");
        rr::require(fs::is_directory(shards) && !fs::is_symlink(shards),
            "Missing original numerical shard directory");
        size_t count = 0;
        for (auto const &entry : fs::recursive_directory_iterator(shards)) {
            rr::require(!entry.is_symlink(), "Symlink in original numerical shard directory");
            if (entry.is_regular_file()) {
                regular_file(entry.path());
                rr::require(entry.path().extension() == ".silo", "Unexpected numerical shard file");
                ++count;
            } else {
                rr::require(entry.is_directory(), "Special file in original numerical shard directory");
            }
        }
        rr::require(count > 0, "Empty original numerical shard directory");
    }
}

void write_new_candidate(fs::path const &output, Json const &candidate)
{
    auto const bytes = candidate.dump(2) + "\n";
    int const descriptor = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    rr::require(descriptor >= 0, "Cannot exclusively create candidate: " + output.string() +
            ": " + std::strerror(errno));
    bool opened = true;
    try {
        size_t written = 0;
        while (written < bytes.size()) {
            auto const count = ::write(descriptor, bytes.data() + written, bytes.size() - written);
            if (count < 0 && errno == EINTR) {
                continue;
            }
            rr::require(count > 0, "Failed to write candidate metadata");
            written += size_t(count);
        }
        rr::require(::fsync(descriptor) == 0, "Failed to sync candidate metadata");
        int const closed = ::close(descriptor);
        opened = false;
        rr::require(closed == 0, "Failed to close candidate metadata");
    } catch (...) {
        if (opened) {
            ::close(descriptor);
        }
        std::error_code ignored;
        fs::remove(output, ignored);
        throw;
    }
}
} // namespace

int main(int argc, char **argv)
{
    try {
        rr::require(argc == 5,
            "Usage: repair_fld_report LEAF CANDIDATE_JSON ORIGINAL_RUN_JSON_SHA256 ORIGINAL_RUN_INI_SHA256");
        auto const requested_leaf = fs::absolute(argv[1]).lexically_normal();
        auto const leaf = fs::canonical(requested_leaf);
        rr::require(leaf == requested_leaf && fs::is_directory(leaf),
            "Original run path must be a real directory without symlinks");
        auto const requested_output = fs::absolute(argv[2]).lexically_normal();
        auto const output_parent = fs::canonical(requested_output.parent_path());
        auto const output = output_parent / requested_output.filename();
        rr::require(!descendant(output, leaf) && !fs::exists(output) && !fs::is_symlink(output),
            "Candidate must be a new file outside the original run directory");
        std::regex const digest("[0-9a-f]{64}");
        std::string const original_metadata_hash = argv[3], original_config_hash = argv[4];
        rr::require(std::regex_match(original_metadata_hash, digest) &&
                std::regex_match(original_config_hash, digest),
            "Original metadata and configuration require explicit SHA256 identities");
        for (auto const *name : {"run.json", "run.ini", "run.log", "radiation-conservation.csv",
                 "momentum-diagnostics.csv"}) {
            regular_file(leaf / name);
        }
        rr::require(rr::sha256(leaf / "run.json") == original_metadata_hash &&
                rr::sha256(leaf / "run.ini") == original_config_hash,
            "Original failed metadata/configuration changed after diagnosis");
        auto const original = rr::read_json(leaf / "run.json");
        validate_identity(leaf, original);
        validate_marker(leaf);
        validate_snapshots(leaf);
        rr::verify_cgs_log(leaf / "run.log", rr::c_cgs);

        auto const diagnostics = rr::read_fld_diagnostics(leaf, original);
        rr::require(diagnostics.at("accepted_steps") == 200 &&
                diagnostics.at("history_start_time") == 0. &&
                diagnostics.at("history_end_time") == stop_time &&
                diagnostics.at("steps").size() == 200,
            "FLD accepted-step history does not cover the complete original run");
        size_t index = 0;
        for (auto const &step : diagnostics.at("steps")) {
            rr::require(step.at("dt") == hard_dt && step.at("t").get<double>() == hard_dt * index,
                "Original FLD macro-step schedule differs");
            ++index;
        }
        rr::read_hydro_profile(leaf, original);
        auto const budget = rr::read_conservation(leaf, original);
        rr::require(bool(budget) && budget->summary.size() == 1 &&
                budget->summary.at(0).at("history_scope") == "full_run" &&
                budget->history.rows.front().at(0) == 0. &&
                budget->history.rows.back().at(0) == stop_time,
            "Missing full original radiation-energy conservation history");

        auto candidate = original;
        candidate["fld_diagnostics"] = diagnostics;
        candidate["conservation"] = budget->summary;
        candidate["status"] = "complete";
        candidate.erase("error");
        auto preserved = candidate;
        preserved.erase("fld_diagnostics");
        preserved.erase("conservation");
        preserved["status"] = original.at("status");
        preserved["error"] = original.at("error");
        rr::require(preserved == original, "Candidate changed original scientific provenance");
        rr::require(rr::sha256(leaf / "run.json") == original_metadata_hash &&
                rr::sha256(leaf / "run.ini") == original_config_hash,
            "Original metadata/configuration changed during validation");
        write_new_candidate(output, candidate);
        std::cout << Json{{"status", "validated_candidate"}, {"candidate", output.string()},
                         {"original_run_metadata_sha256", original_metadata_hash},
                         {"original_config_sha256", original_config_hash},
                         {"candidate_sha256", rr::sha256(output)},
                         {"accepted_steps", diagnostics.at("accepted_steps")},
                         {"history_start_time", diagnostics.at("history_start_time")},
                         {"history_end_time", diagnostics.at("history_end_time")},
                         {"source_snapshots", 121}, {"solver_invoked", false},
                         {"original_run_modified", false}}
                         .dump(2)
                  << '\n';
        return 0;
    } catch (std::exception const &error) {
        std::cerr << "FLD reporting recovery refused: " << error.what() << '\n';
        return 1;
    }
}
