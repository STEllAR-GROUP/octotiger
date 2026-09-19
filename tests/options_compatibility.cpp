#include "octotiger/options_compatibility.hpp"

#include <boost/program_options.hpp>

#include <cmath>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace po = boost::program_options;
using namespace octotiger::options_compatibility;

namespace {

struct values {
    bool radiation = false;
    double opacity = -1.0;
    double gamma = 1.4;
    bool gravity = true;
    std::string problem = "NONE";
};

struct fixture {
    values value;
    po::options_description canonical{"Canonical options"};
    po::options_description legacy{"Legacy options"};
    po::options_description all{"All options"};
    std::vector<migration> migrations;

    fixture() {
        legacy.add_options()
            ("radiation", po::value<bool>(&value.radiation)->default_value(false))
            ("rad_opacity", po::value<double>(&value.opacity)->default_value(-1.0))
            ("sod_gamma", po::value<double>(&value.gamma)->default_value(1.4))
            ("gravity", po::value<bool>(&value.gravity)->default_value(true))
            ("problem", po::value<std::string>(&value.problem)->default_value("NONE"));
        add_canonical_option(canonical, migrations,
            "radiation.enabled", "radiation", &value.radiation);
        add_canonical_option(canonical, migrations,
            "radiation.opacity.constant", "rad_opacity", &value.opacity);
        add_canonical_option(canonical, migrations,
            "hydro.gamma", "sod_gamma", &value.gamma);
        add_canonical_option(canonical, migrations,
            "gravity.enabled", "gravity", &value.gravity);
        add_canonical_option(canonical, migrations,
            "problem.name", "problem", &value.problem);
        all.add(canonical).add(legacy);
    }

    bool parse(std::vector<std::string> const& cli, std::string const& config = {},
        std::string* diagnostics = nullptr) {
        po::variables_map map;
        std::set<std::string> supplied;
        auto const command_line = po::command_line_parser(cli).options(all).run();
        remember_supplied(supplied, command_line);
        po::store(command_line, map);
        po::notify(map);
        reapply_canonical_values(map, canonical, migrations);
        if (!config.empty()) {
            std::istringstream input(config);
            auto const file = po::parse_config_file(input, all);
            remember_supplied(supplied, file);
            po::store(file, map);
        }
        std::ostringstream messages;
        if (!check_compatibility_spellings(supplied, migrations, messages)) {
            if (diagnostics != nullptr) {
                *diagnostics = messages.str();
            }
            return false;
        }
        po::notify(map);
        reapply_canonical_values(map, canonical, migrations);
        warn_legacy_spellings(supplied, migrations, messages);
        if (diagnostics != nullptr) {
            *diagnostics = messages.str();
        }
        return true;
    }
};

int failures = 0;

void check(bool condition, char const* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    {
        fixture test;
        check(test.parse({}), "defaults parse");
        check(!test.value.radiation && test.value.gravity && test.value.gamma == 1.4,
            "historical defaults");
    }
    {
        fixture test;
        check(test.parse({"--radiation.enabled=true", "--hydro.gamma=1.6",
                  "--gravity.enabled=false", "--problem.name=BLAST"}),
            "canonical parse");
        check(test.value.radiation && !test.value.gravity && test.value.gamma == 1.6 &&
                test.value.problem == "BLAST",
            "canonical values");
    }
    {
        fixture test;
        std::string diagnostics;
        check(test.parse({"--radiation=true", "--sod_gamma=1.5"}, {}, &diagnostics),
            "legacy parse");
        check(test.value.radiation && test.value.gamma == 1.5, "legacy values");
        check(diagnostics.find("option 'radiation' is deprecated") != std::string::npos,
            "legacy radiation warning");
        check(diagnostics.find("option 'sod_gamma' is deprecated") != std::string::npos,
            "legacy hydro warning");
    }
    {
        fixture test;
        check(test.parse({"--radiation.enabled=true", "--hydro.gamma=1.8"},
                  "radiation.enabled=false\nhydro.gamma=1.2\n"),
            "config and CLI parse");
        check(test.value.radiation && test.value.gamma == 1.8,
            "command line overrides config");
    }
    {
        fixture test;
        std::string diagnostics;
        check(!test.parse({"--radiation.enabled=true"}, "radiation=false\n", &diagnostics),
            "mixed-spelling conflict");
        check(diagnostics.find("both legacy option 'radiation'") != std::string::npos,
            "conflict explains spellings");
    }
    {
        fixture original;
        check(original.parse({"--radiation.enabled=true", "--radiation.opacity.constant=2.5",
                  "--hydro.gamma=1.7", "--gravity.enabled=false", "--problem.name=SOD"}),
            "round-trip source parse");
        std::ostringstream serialized;
        serialized << std::boolalpha
                   << "radiation.enabled=" << original.value.radiation << '\n'
                   << "radiation.opacity.constant=" << original.value.opacity << '\n'
                   << "hydro.gamma=" << original.value.gamma << '\n'
                   << "gravity.enabled=" << original.value.gravity << '\n'
                   << "problem.name=" << original.value.problem << '\n';
        fixture restored;
        check(restored.parse({}, serialized.str()), "round-trip config parse");
        check(restored.value.radiation == original.value.radiation &&
                restored.value.opacity == original.value.opacity &&
                restored.value.gamma == original.value.gamma &&
                restored.value.gravity == original.value.gravity &&
                restored.value.problem == original.value.problem,
            "round-trip values");
    }
    return failures == 0 ? 0 : 1;
}
