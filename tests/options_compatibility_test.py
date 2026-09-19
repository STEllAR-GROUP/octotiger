#!/usr/bin/env python3
"""Source-level contract tests for the Step 01 options compatibility layer.

These tests intentionally need neither HPX nor Silo, so distributions can run
the compatibility checks even when the full application dependencies are not
available.  The supported application builds additionally compile the parser.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROCESSING = (ROOT / "src/options_processing.cpp").read_text()
HEADER = (ROOT / "octotiger/options.hpp").read_text()
COMPATIBILITY = (ROOT / "octotiger/options_compatibility.hpp").read_text()


def mappings():
    begin = PROCESSING.index("po::options_description canonical_opts")
    end = PROCESSING.index("#undef CANONICAL_MULTI", begin)
    block = PROCESSING[begin:end]
    result = {
        legacy: canonical
        for canonical, legacy in re.findall(
            r'(?:CANONICAL|CANONICAL_MULTI)\("([^"]+)", "([^"]+)"', block
        )
    }
    result["help"] = "runtime.help"
    return result


def legacy_names():
    begin = PROCESSING.index("legacy_opts.add_options()")
    end = PROCESSING.index("po::options_description canonical_opts", begin)
    block = PROCESSING[begin:end]
    # Registrations are followed by either po::value or their flag description.
    # This avoids string-valued defaults such as "thermal" and "./".
    return set(re.findall(r'\("([^" ]+)"\s*,\s*(?:po::value|\")', block))


class OptionsCompatibility(unittest.TestCase):
    def test_every_legacy_parser_name_has_one_canonical_name(self):
        table = mappings()
        self.assertEqual(legacy_names(), set(table))
        self.assertEqual(len(table.values()), len(set(table.values())))

    def test_representative_domain_migrations(self):
        table = mappings()
        self.assertEqual(table["radiation"], "radiation.enabled")
        self.assertEqual(table["rad_opacity"], "radiation.opacity.constant")
        self.assertEqual(table["sod_gamma"], "hydro.gamma")
        self.assertEqual(table["gravity"], "gravity.enabled")
        self.assertEqual(table["problem"], "problem.name")
        self.assertEqual(table["eblast0"], "blast.energy")

    def test_historical_defaults_are_unchanged(self):
        expected = {
            "radiation": "false",
            "rad_opacity": "-1.0",
            "sod_gamma": "1.4",
            "gravity": "true",
            "problem": "NONE",
        }
        for name, value in expected.items():
            pattern = rf'\("{name}".*?default_value\({re.escape(value)}\)'
            self.assertRegex(PROCESSING, pattern)

    def test_precedence_and_ambiguity_contract_is_wired(self):
        cli = PROCESSING.index("po::store(command_line, vm)")
        config = PROCESSING.index("po::store(config, vm)")
        final_notify = PROCESSING.index("po::notify(vm)", config)
        conflict = PROCESSING.index("check_compatibility_spellings", config)
        self.assertLess(cli, config)
        self.assertLess(config, conflict)
        self.assertLess(conflict, final_notify)
        self.assertIn("supplied.count(legacy)", COMPATIBILITY)
        self.assertIn("supplied.count(canonical)", COMPATIBILITY)
        self.assertIn("warned.insert(entry.first).second", COMPATIBILITY)

    def test_legacy_help_warns_before_returning(self):
        help_branch = PROCESSING.index(
            'if (vm.count("help") || vm.count("runtime.help"))'
        )
        warning = PROCESSING.index("warn_legacy_spellings", help_branch)
        help_return = PROCESSING.index("return false", warning)
        self.assertLess(help_branch, warning)
        self.assertLess(warning, help_return)

    def test_nested_views_alias_serialized_storage(self):
        for group in (
            "radiation_options",
            "hydro_options",
            "gravity_options",
            "problem_options",
            "output_options",
            "mesh_options",
            "restart_options",
            "units_options",
            "blast_options",
            "timestep_options",
            "runtime_options",
            "execution_options",
        ):
            self.assertIn(group, HEADER)
        self.assertIn("enabled(owner.radiation)", HEADER)
        self.assertIn("gamma(owner.sod_gamma)", HEADER)
        self.assertIn("filename(owner.restart_filename)", HEADER)

    def test_round_trip_storage_remains_in_historical_archive(self):
        # The hierarchy is a view: canonical group objects must not enter the
        # archive, while representative legacy storage fields remain present.
        serialize = HEADER[HEADER.index("void serialize"):]
        for field in (
            "radiation",
            "rad_opacity",
            "sod_gamma",
            "gravity",
            "output_filename",
        ):
            self.assertRegex(serialize, rf'arc\s*&\s*{field}\b')
        self.assertIn("int tmp = problem", serialize)
        self.assertIn("problem = static_cast<problem_type>(tmp)", serialize)
        self.assertNotRegex(serialize, r'arc\s*&\s*(radiation|hydro|gravity)_options\b')


if __name__ == "__main__":
    unittest.main()
