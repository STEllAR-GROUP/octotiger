#!/usr/bin/env python3
"""Source-level contract tests for the Step 01 options compatibility layer.

These tests intentionally need neither HPX nor Silo, so distributions can run
the compatibility checks even when the full application dependencies are not
available.  The supported application builds additionally compile the parser.
"""

from pathlib import Path
import re
import unittest


root = Path(__file__).resolve().parents[1]
processing = (root / "src/options_processing.cpp").read_text()
header = (root / "octotiger/options.hpp").read_text()
compatibility = (root / "octotiger/options_compatibility.hpp").read_text()
greyOpacity = (
    root / "octotiger/radiation/grey_opacity_options.hpp"
).read_text()


def mappings():
    begin = processing.index("po::options_description canonicalOptions")
    end = processing.index("#undef canonicalMultiOption", begin)
    block = processing[begin:end]
    result = {
        legacy: canonical
        for canonical, legacy in re.findall(
            r'(?:canonicalOption|canonicalMultiOption)\("([^"]+)", "([^"]+)"', block
        )
    }
    return result


def legacyNames():
    begin = processing.index("legacyOptions.add_options()")
    end = processing.index("po::options_description canonicalOptions", begin)
    block = processing[begin:end]
    # Registrations are followed by either po::value or their flag description.
    # This avoids string-valued defaults such as "thermal" and "./".
    return set(re.findall(r'\(\s*"([^" ]+)"\s*,\s*(?:po::value|\")', block))


class OptionsCompatibility(unittest.TestCase):
    def test_every_legacy_parser_name_has_one_canonical_name(self):
        table = mappings()
        obsolete = {"correct_am_hydro", "X", "Z"}
        self.assertEqual(legacyNames() - obsolete, set(table))
        self.assertEqual(len(table.values()), len(set(table.values())))

    def test_obsolete_angular_momentum_correction_is_disabled(self):
        self.assertIn("if (opts().correct_am_hydro)", processing)

    def test_obsolete_composition_options_are_rejected(self):
        table = mappings()
        self.assertNotIn("X", table)
        self.assertNotIn("Z", table)
        self.assertNotIn("hydro.species.hydrogen_fraction", processing)
        self.assertNotIn("hydro.species.metallicity", processing)
        self.assertIn('for (auto const* option : {"X", "Z"})', processing)
        self.assertIn("is obsolete and unsupported", processing)
        self.assertIn("hydro.species.atomic_mass", processing)
        self.assertIn("hydro.species.atomic_number", processing)

    def test_representative_domain_migrations(self):
        table = mappings()
        self.assertEqual(table["radiation"], "radiation.enabled")
        self.assertEqual(table["rad_opacity"], "radiation.opacity.constant")
        self.assertEqual(table["sod_gamma"], "hydro.gamma")
        self.assertEqual(table["gravity"], "gravity.enabled")
        self.assertEqual(table["problem"], "problem.name")
        self.assertEqual(table["eblast0"], "problem.blast.energy")
        self.assertEqual(table["ndim"], "mesh.ndim")

    def test_mesh_dimension_contract(self):
        self.assertRegex(processing, r'"ndim"[\s\S]*?default_value\(3\)')
        self.assertIn("opts().dimensionCount < 1", processing)
        self.assertIn("opts().dimensionCount > 3", processing)
        self.assertIn("gravity is only supported with mesh.ndim=3", processing)
        self.assertNotIn("radiation transport currently requires mesh.ndim=3", processing)

    def test_refinement_is_a_top_level_group(self):
        table = mappings()
        expected = {
            "accretor_refine": "refinement.accretor_levels",
            "core_refine": "refinement.core",
            "refinement_floor": "refinement.density_floor",
            "grad_rho_refine": "refinement.density_gradient",
            "donor_refine": "refinement.donor_levels",
        }
        for legacy, canonical in expected.items():
            self.assertEqual(table[legacy], canonical)
        self.assertNotIn("mesh.refinement.", processing)

    def test_historical_defaults_are_unchanged(self):
        expected = {
            "radiation": "false",
            "rad_opacity": "-1.0",
            "sod_gamma": "1.4",
            "gravity": "true",
            "problem": "NONE",
        }
        for name, value in expected.items():
            pattern = rf'\(\s*"{name}"[\s\S]*?default_value\({re.escape(value)}\)'
            self.assertRegex(processing, pattern)

    def test_precedence_and_ambiguity_contract_is_wired(self):
        cli = processing.index("po::store(commandLine, vm)")
        config = processing.index("po::store(config, vm)")
        finalNotify = processing.index("po::notify(vm)", config)
        conflict = processing.index("checkCompatibilitySpellings", config)
        self.assertLess(cli, config)
        self.assertLess(config, conflict)
        self.assertLess(conflict, finalNotify)
        self.assertIn("supplied.count(legacy)", compatibility)
        self.assertIn("supplied.count(canonical)", compatibility)
        self.assertIn("seen.insert(entry.first).second", compatibility)

    def test_legacy_aliases_share_one_migration_warning(self):
        self.assertEqual(
            compatibility.count(
                "WARNING: legacy options were used. These options have been upgraded"
            ),
            1,
        )
        self.assertIn('"  --" << entry.first << " -> --" << entry.second', compatibility)
        self.assertNotIn("is deprecated; use", compatibility)

    def test_canonical_help_reuses_descriptive_option_text(self):
        self.assertIn("option->description()", compatibility)
        self.assertIn("legacy spelling: --", compatibility)
        self.assertNotIn("canonical spelling; legacy", compatibility)
        self.assertIn("legacyOptions, migrations", processing)

    def test_effective_option_report_uses_canonical_names_and_sources(self):
        self.assertNotIn("#define SHOW", processing)
        self.assertIn("std::vector<OptionDisplay> displayedOptions", processing)
        self.assertIn("{canonical, legacy, []() { return to_string(opts().member); }}", processing)
        self.assertIn("option.canonical", processing)
        self.assertIn("option.legacy", processing)
        self.assertIn("[specified]", processing)
        self.assertIn("[default]", processing)
        self.assertIn("Effective canonical options", processing)
        self.assertIn("to_string(std::vector<T> const& values)", processing)
        self.assertIn("to_string(const int& num)", processing)
        self.assertIn("output << '['", processing)
        self.assertIn('output << ", "', processing)
        self.assertIn("radiation.opacity.transport_absorption", processing)

        configStore = processing.index("po::store(config, vm)")
        report = processing.index("showEffectiveOptions();")
        normalize = processing.index("normalize_constants();")
        self.assertLess(configStore, report)
        self.assertLess(normalize, report)

    def test_test_ini_files_use_only_registered_hierarchical_names(self):
        paths = sorted((root / "test_problems").rglob("*.ini"))
        paths += sorted((root / "verification_results" / "radiation" / "configs").glob("*.ini"))
        self.assertTrue(paths)
        legacy = legacyNames()
        canonical = set(mappings().values())
        canonical.update(re.findall(r'\("([a-z][a-z0-9_.]+)"', greyOpacity))
        failures = []
        for path in paths:
            for lineNumber, raw in enumerate(path.read_text().splitlines(), 1):
                match = re.match(r"\s*[#;]?\s*([^#;\s=]+)\s*=", raw)
                if match is None:
                    continue
                key = match.group(1)
                if key in legacy or key not in canonical:
                    failures.append(f"{path.relative_to(root)}:{lineNumber}: {key}")
        self.assertEqual(failures, [])

    def test_help_is_canonical_and_unchanged(self):
        self.assertNotIn("runtime.help", processing)
        self.assertNotIn('migrations.emplace_back("help"', processing)
        self.assertNotIn("help", legacyNames())

        helpBranch = processing.index('if (vm.count("help"))')
        warning = processing.index("warnLegacySpellings", helpBranch)
        helpReturn = processing.index("return false", warning)
        self.assertLess(helpBranch, warning)
        self.assertLess(warning, helpReturn)

    def test_nested_views_alias_serialized_storage(self):
        for group in (
            "radiationOptions",
            "hydroOptions",
            "gravityOptions",
            "problemOptions",
            "outputOptions",
            "meshOptions",
            "refinementOptions",
            "restartOptions",
            "unitsOptions",
            "blastOptions",
            "timestepOptions",
            "runtimeOptions",
            "executionOptions",
        ):
            self.assertIn(group, header)
        self.assertIn("enabled(owner.radiation)", header)
        self.assertIn("gamma(owner.sod_gamma)", header)
        self.assertIn("densityGradient(owner.grad_rho_refine)", header)
        self.assertIn("filename(owner.restart_filename)", header)

    def test_round_trip_storage_remains_in_historical_archive(self):
        # The hierarchy is a view: canonical group objects must not enter the
        # archive, while representative legacy storage fields remain present.
        serialize = header[header.index("void serialize"):]
        for field in (
            "radiation",
            "radOpacity",
            "sod_gamma",
            "gravity",
            "output_filename",
        ):
            self.assertRegex(serialize, rf'arc\s*&\s*{field}\b')
        self.assertIn("int tmp = problem", serialize)
        self.assertIn("problem = static_cast<problem_type>(tmp)", serialize)
        self.assertNotRegex(serialize, r'arc\s*&\s*(radiation|hydro|gravity)Options\b')


if __name__ == "__main__":
    unittest.main()
