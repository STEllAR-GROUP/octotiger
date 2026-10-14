import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import momentum_diagnostics as md


def metadata(model="m1", case="subcritical_shock"):
    return {"case": case, "level": 0, "radiation_model": model,
            "radiation_reduced_c_factor": 1, "time": 2, "length": 2,
            "config": {"periodic": "off", "code_to_g": "1", "code_to_cm": "1", "code_to_s": "1"}}


def document():
    return {"schema": 1, "integration": "full-volume leaf-cell", "units": "cgs",
            "precision": "runtime long-double reduction", "records": [
                {"time": 0, "physical_c": 10, "volume": 8, "gas_energy": 80, "radiation_energy": 20,
                 "gas_momentum": [3, 0, 0], "radiation_momentum": [2, 0, 0],
                 "gas_boundary_momentum": [0, 0, 0], "radiation_boundary_momentum": [0, 0, 0],
                 "external_impulse": [0, 0, 0]},
                {"time": 2, "physical_c": 10, "volume": 8, "gas_energy": 90, "radiation_energy": 10,
                 "gas_momentum": [7, -1, 0], "radiation_momentum": [1, 2, 0],
                 "gas_boundary_momentum": [-2, -1, 0], "radiation_boundary_momentum": [-1, 0, 0],
                 "external_impulse": [0, 0, 0]}]}


class MomentumTests(unittest.TestCase):
    def test_open_boundary_balance_and_scale(self):
        result = md.build_diagnostic(document(), metadata())
        self.assertEqual(result["normalization"]["value"], 10)
        self.assertEqual(result["normalization"]["momentum_to_normalization_ratio"], .5)
        self.assertIn("amplify floating-point roundoff", " ".join(result["notes"]))
        final = result["records"][-1]
        self.assertEqual(final["raw_delta"], [3, 1, 0])
        self.assertEqual(final["raw_delta_normalized"], [.3, .1, 0])
        self.assertEqual(final["boundary_corrected_delta"], [0, 0, 0])
        self.assertEqual(final["accounted_balance"], [0, 0, 0])
        self.assertAlmostEqual(final["raw_delta_magnitude"], 10 ** .5)

    def test_decimal_large_baseline(self):
        doc = document()
        doc["records"][0]["gas_momentum"][0] = "1000000000000000000000000000000"
        doc["records"][1]["gas_momentum"][0] = "1000000000000000000000000000001"
        doc["records"][1]["radiation_momentum"][0] = 2
        self.assertEqual(md.build_diagnostic(doc, metadata())["records"][-1]["raw_delta"][0], 1)

    def test_fld_missing_boundary_never_zero(self):
        doc = document()
        for record in doc["records"]:
            del record["radiation_boundary_momentum"]
        result = md.build_diagnostic(doc, metadata("fld"))
        self.assertIsNone(result["records"][-1]["boundary_corrected_delta"])
        self.assertTrue(result["available"]["raw_delta"])
        self.assertFalse(result["available"]["accounted_balance"])
        self.assertIn("not an exactly conserved", " ".join(result["notes"]))

    def test_periodic_boundaries_known(self):
        doc = document()
        for record in doc["records"]:
            del record["radiation_boundary_momentum"]
            del record["gas_boundary_momentum"]
        meta = metadata("fld")
        meta["config"]["periodic"] = "on"
        result = md.build_diagnostic(doc, meta)
        self.assertEqual(result["records"][-1]["boundary_corrected_delta"], [3, 1, 0])

    def test_fixed_material_omitted(self):
        docs = [md.build_diagnostic(document(), metadata(model, "gaussian_pulse")) for model in ("m1", "fld")]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "should-not-exist"
            result = md.write_pair(docs, output)
            self.assertTrue(result["omitted"])
            self.assertEqual(result["plots"], [])
            self.assertFalse(output.exists())

    def test_internal_exchange_not_subtracted(self):
        doc = document()
        for record in doc["records"]:
            del record["external_impulse"]
        result = md.build_diagnostic(doc, metadata())
        self.assertEqual(result["records"][-1]["accounted_balance"], [0, 0, 0])

    def test_reduced_c_warning(self):
        meta = metadata()
        meta["radiation_reduced_c_factor"] = .1
        self.assertIn("weighted momentum", " ".join(md.build_diagnostic(document(), meta)["notes"]))

    def test_invalid_histories(self):
        for field, value in (("time", 0), ("physical_c", 0), ("gas_energy", -1), ("volume", 0),
                             ("radiation_momentum", [1, 2])):
            doc = document()
            doc["records"][-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                md.build_diagnostic(doc, metadata())

    def test_restart_does_not_rename_segment_energy_initial(self):
        doc = document()
        doc["records"][0]["time"] = 1
        result = md.build_diagnostic(doc, metadata())
        self.assertIsNone(result["normalization"]["value"])
        self.assertIsNone(result["records"][-1]["raw_delta_normalized"])

    def test_paired_artifacts_synthetic_only(self):
        m1 = md.build_diagnostic(document(), metadata())
        fld_doc = document()
        for record in fld_doc["records"]:
            del record["radiation_boundary_momentum"]
        fld = md.build_diagnostic(fld_doc, metadata("fld"))
        with tempfile.TemporaryDirectory() as folder:
            result = md.write_pair([fld, m1], Path(folder))
            self.assertEqual(result["models"], ["m1", "fld"])
            page = (Path(folder) / "index.html").read_text()
            self.assertIn("M1 RK1", page)
            self.assertIn("FLD", page)
            svg = (Path(folder) / "boundary-corrected-delta.svg").read_text()
            self.assertIn("Matplotlib", svg)
            self.assertIn("FLD (unavailable", svg)
            self.assertNotIn("nan", svg.lower())
            with (Path(folder) / "fld-momentum.csv").open() as stream:
                final = list(csv.DictReader(stream))[-1]
            self.assertEqual(final["boundary_corrected_delta_x"], "")
            self.assertEqual(final["raw_delta_x"], "3.0")
            json.loads((Path(folder) / "fld-momentum.json").read_text())

    def test_pair_mismatch(self):
        m1 = md.build_diagnostic(document(), metadata())
        fld = md.build_diagnostic(document(), metadata("fld", "supercritical_shock"))
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            md.write_pair([m1, fld], Path(folder))

    def test_runtime_blanks_and_source(self):
        columns = ("time,physical_c,volume,gas_energy,radiation_energy,gas_px,gas_py,gas_pz,"
                   "radiation_px,radiation_py,radiation_pz,gas_px_boundary,gas_py_boundary,gas_pz_boundary,"
                   "radiation_boundary_available,radiation_px_boundary,radiation_py_boundary,radiation_pz_boundary,"
                   "external_impulse_available,external_px,external_py,external_pz,units\n")
        rows = "0,10,8,80,20,3,0,0,2,0,0,0,0,0,0,,,,1,0,0,0,cgs\n2,10,8,90,10,7,-1,0,1,2,0,-2,-1,0,0,,,,1,0,0,0,cgs\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "momentum-diagnostics.csv"
            path.write_text(columns + rows)
            result = md.build_diagnostic(None, metadata("fld"), total_momentum_csv_path=path)
            self.assertIsNone(result["records"][-1]["boundary_corrected_delta"])
            self.assertEqual(result["records"][-1]["raw_delta"], [3, 1, 0])
            path.write_text(columns + rows.replace(",0,,,,1,", ",0,0,0,0,1,"))
            with self.assertRaises(ValueError):
                md.read_runtime(path)


if __name__ == "__main__":
    unittest.main()
