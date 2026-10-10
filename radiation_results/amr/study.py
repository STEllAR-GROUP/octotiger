#!/usr/bin/env python3
"""Offline initial-profile threshold proposals and bounded local AMR smoke runs.

No remote access, full-duration solves, quota-based tagging, or publication.
Threshold proposals measure initial occupancy, not solution accuracy.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
CASES = ("streaming_front", "streaming_wave", "gaussian_pulse", "equilibrium_sphere",
         "subcritical_shock", "supercritical_shock", "radiating_blast")
COUPLED = CASES[4:]
C = 2.99792458e10


def limited_environment():
    result = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        result[name] = "1"
    return result


def ratio(base_blocks, refined_blocks):
    coarse = base_blocks - refined_blocks
    return 8 * refined_blocks / coarse if coarse else None


def occupancy(base_blocks, refined_blocks):
    return dict(refined_base_blocks=refined_blocks, coarse_leaf_blocks=base_blocks-refined_blocks,
                fine_leaf_blocks=8*refined_blocks,
                fine_to_coarse_leaf_cell_ratio=ratio(base_blocks, refined_blocks),
                fine_volume_fraction=refined_blocks/base_blocks)


def threshold_candidates(scores):
    """Keep symmetry-related scores together; do not select arbitrary blocks.

    Midpoints between separated score groups provide robust scalar thresholds.
    Groups within 1e-9 of the global peak are inseparable for this study, so
    floating-point roundoff cannot be used to break geometric symmetry.
    """
    ordered = sorted(scores, reverse=True)
    if not ordered or ordered[0] <= 0:
        return []
    tolerance = 1e-9 * ordered[0]
    groups = []
    for value in ordered:
        if value <= tolerance:
            continue
        if groups and groups[-1][-1] - value <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    result, count = [], 0
    for index, group in enumerate(groups):
        count += len(group)
        next_value = groups[index+1][0] if index+1 < len(groups) else 0
        threshold = .5 * (group[-1] + next_value)
        result.append(dict(threshold=threshold, **occupancy(len(scores), count)))
    return result


def summarize_sample(sample, mode, low=.1, high=1.):
    candidates = threshold_candidates([values[mode == "relative"] for values in sample["block_scores"]])
    feasible = [row for row in candidates if row["fine_to_coarse_leaf_cell_ratio"] is not None
                and low <= row["fine_to_coarse_leaf_cell_ratio"] <= high]
    if feasible:
        target = math.sqrt(low*high)
        selected = min(feasible, key=lambda row: abs(math.log(row["fine_to_coarse_leaf_cell_ratio"]/target)))
        status = "initial_ratio_feasible"
    elif candidates:
        # A failed occupancy target must remain visible. The user permits all
        # fine cells for coarse tests. At higher levels retain the complete
        # strongest-gradient tie group, never a quota-selected subset of it.
        selected = candidates[-1] if sample["finest_level"] <= 3 else candidates[0]
        status = "all_fine_initial" if selected["coarse_leaf_blocks"] == 0 else "initial_ratio_unattainable"
    else:
        selected = dict(threshold=1., **occupancy(sample["base_blocks"], 0))
        status = "no_initial_gradient"
    result = {key: value for key, value in sample.items() if key != "block_scores"}
    result.update(mode=mode, status=status, selected=selected, attainable=candidates,
                  minimum_nonempty=candidates[0] if candidates else None)
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def prepare(args):
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "sample-initial"
    command = [args.cxx, "-std=c++20", "-O2", "-I"+str(ROOT),
               str(Path(__file__).with_name("sample_initial.cpp")), "-o", str(binary)]
    subprocess.run(command, check=True, env=limited_environment(), timeout=60)
    rows = []
    for case in args.cases:
        field = "density" if case == "radiating_blast" and args.blast_density else "radiation_energy"
        for level in args.levels:
            started = time.monotonic()
            output = subprocess.check_output([str(binary), case, str(level), field, str(args.floor), str(args.buffer)],
                                             env=limited_environment(), timeout=args.sample_timeout)
            sample = json.loads(output)
            write_json(folder / (case+"-l%d-sample.json" % level), sample)
            row = summarize_sample(sample, args.mode)
            row["sample_seconds"] = time.monotonic()-started
            rows.append(row)
            write_json(folder / "plan.json", dict(scope="Initial-profile occupancy proposals; no AMR evolution or accuracy validation",
                       ratio_target=[.1,1.], source_root=str(ROOT), mode=args.mode, buffer=args.buffer,
                       gaussian_note="t=0 exact periodized cell integrals; reference-generator FFT roundoff is not reproduced",
                       source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (ROOT/"octotiger/amr/gradient.hpp", ROOT/"test_problems/radiation/profiles.hpp",
                                        ROOT/"test_problems/radiation/hydro_benchmarks.hpp")}, rows=rows))
            selected = row["selected"]
            print("%s L%d %s threshold=%.9g R=%s" % (case, level, row["status"], selected["threshold"],
                   selected["fine_to_coarse_leaf_cell_ratio"]), flush=True)


def read_config(path):
    result = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#",1)[0].strip()
        if line:
            key, value = line.split("=",1)
            result[key.strip()] = value.strip()
    return result


def read_histogram(path, finest_level, allow_all_fine_through=3):
    groups = {}
    with path.open() as stream:
        reader = csv.DictReader(stream)
        required = {"t", "step", "level", "leaf_blocks", "leaf_cells"}
        if not required.issubset(reader.fieldnames or ()):
            raise AssertionError("Missing AMR histogram columns")
        for row in reader:
            t = float(row["t"])
            step, level, blocks, cells = (int(row[name]) for name in ("step", "level", "leaf_blocks", "leaf_cells"))
            if not math.isfinite(t) or t < 0 or step < 0 or blocks < 0 or cells != 512*blocks:
                raise AssertionError("Invalid AMR histogram row")
            if blocks and level not in (finest_level-1, finest_level):
                raise AssertionError("Active leaf lies outside the configured adjacent level pair")
            key = (step,t)
            if level in groups.setdefault(key, {}):
                raise AssertionError("Duplicate level in AMR histogram sample")
            groups[key][level] = (blocks,cells)
    if not groups or not any(t > 0 for step,t in groups):
        raise AssertionError("No evolved AMR histogram sample; run through at least one regrid")
    summaries = []
    for (step,t), levels in sorted(groups.items()):
        cb, cc = levels.get(finest_level-1,(0,0))
        fb, fc = levels.get(finest_level,(0,0))
        if cb+fb/8 != 8**(finest_level-1):
            raise AssertionError("AMR leaf counts do not cover the domain exactly once")
        if fb % 8:
            raise AssertionError("Refinement must create complete eight-child families")
        value = fc/cc if cc else None
        if cc == 0:
            status = "all_fine_coarse_exception" if finest_level <= allow_all_fine_through else "all_fine_outside_target"
        elif fc == 0:
            status = "all_coarse_outside_target"
        elif .1 <= value <= 1:
            status = "ratio_in_target"
        else:
            status = "mixed_outside_target"
        summaries.append(dict(step=step, t=t, coarse_leaf_blocks=cb, fine_leaf_blocks=fb,
                              coarse_leaf_cells=cc, fine_leaf_cells=fc,
                              fine_to_coarse_leaf_cell_ratio=value, status=status))
    return summaries


def check_ledger(path):
    with path.open() as stream:
        rows = [{key:float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    if len(rows) < 2 or rows[-1]["t"] <= rows[0]["t"]:
        raise AssertionError("No evolved radiation conservation sample")
    if any(not math.isfinite(value) for row in rows for value in row.values()):
        raise AssertionError("Nonfinite radiation conservation data")
    initial = rows[0]
    worst = 0
    for row in rows[1:]:
        source = row["er_source"]-initial["er_source"]
        boundary = row["er_boundary"]-initial["er_boundary"]
        scale = max(abs(row["er"]), abs(initial["er"]), abs(source), abs(boundary), 1e-300)
        residual = abs(row["er"]-initial["er"]+boundary-source)/scale
        worst = max(worst,residual)
    if worst > 1e-9:
        raise AssertionError("AMR radiation ledger does not close: %g" % worst)
    return dict(samples=len(rows), worst_relative_energy_residual=worst, final_time=rows[-1]["t"])


def run_bounded(command, folder, log, timeout, memory_limit_gib):
    """Bound the local child and retain measured RSS; never launch remotely."""
    available = None
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1])*1024
    requested_limit = memory_limit_gib*1024**3
    limit = min(requested_limit,.5*available) if available else requested_limit
    started, peak_rss = time.monotonic(), 0
    process = subprocess.Popen(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,
                               env=limited_environment(),start_new_session=True)
    try:
        while process.poll() is None:
            status = Path("/proc")/str(process.pid)/"status"
            try:
                for line in status.read_text().splitlines():
                    if line.startswith(("VmRSS:","VmHWM:")):
                        peak_rss = max(peak_rss,int(line.split()[1])*1024)
            except FileNotFoundError:
                pass
            if peak_rss > limit:
                raise RuntimeError("Local process exceeded bounded RSS budget: %.3f GiB > %.3f GiB" %
                                   (peak_rss/1024**3,limit/1024**3))
            remaining = timeout-(time.monotonic()-started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command,timeout)
            try:
                process.wait(timeout=min(.25,remaining))
            except subprocess.TimeoutExpired:
                pass
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode,command)
        return dict(sampled_peak_rss_gib=peak_rss/1024**3,memory_budget_gib=limit/1024**3,
                    wall_seconds=time.monotonic()-started)
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()


def partial_evidence(folder):
    """Record available observations after failure without certifying the run."""
    result = {"scope":"Incomplete observations only; output buffering can hide the last steps"}
    histogram = folder/"gradient-amr.csv"
    if histogram.exists():
        with histogram.open() as stream:
            rows = list(csv.DictReader(stream))
        result["recorded_mesh_rows"] = rows
        result["has_post_regrid_mesh_sample"] = any(float(row["t"]) > 0 for row in rows)
    ledger = folder/"radiation-conservation.csv"
    if ledger.exists():
        with ledger.open() as stream:
            rows = list(csv.DictReader(stream))
        result["last_recorded_conservation_time"] = float(rows[-1]["t"]) if rows else None
    log = folder/"run.log"
    if log.exists():
        steps = re.findall(r"^TS\s+(\d+)::\s+t:\s+([^,\s]+)",log.read_text(errors="replace"),re.MULTILINE)
        result["last_recorded_step"] = int(steps[-1][0]) if steps else None
        result["last_recorded_step_time"] = float(steps[-1][1]) if steps else None
    return result


def run(args):
    plan = json.loads(args.plan.read_text())
    rows = [row for row in plan["rows"] if row["case"] in args.cases and row["finest_level"] in args.levels]
    expected = {(case,level) for case in args.cases for level in args.levels}
    if {(row["case"],row["finest_level"]) for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("Plan does not contain exactly the requested case/level pairs")
    if args.steps < 6:
        raise ValueError("At least six steps are required to exercise the five-step regrid interval")
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    reports = []
    for row in rows:
        case, level = row["case"], row["finest_level"]
        folder = output/(case+"-l%d" % level)
        folder.mkdir()  # Refuse to overwrite any previous run, including failures.
        config = read_config(ROOT/"radiation_results/configs"/(case+".ini"))
        config.update(unigrid="off", min_level=level-1, max_level=level, stop_step=args.steps-1,
                      disable_analytic="on", disable_output="on", disable_diagnostics="on", odt="1e99",
                      datadir=str(folder)+os.sep, gradient_refine_field=row["field"],
                      gradient_refine_mode=row["mode"], gradient_refine_threshold=row["selected"]["threshold"],
                      gradient_refine_floor=row["floor"], gradient_refine_buffer=row["buffer"],
                      gradient_refine_coarsen_factor=.8, radiation_reduced_c_factor=1,
                      silo_num_groups=1)
        if case not in COUPLED:
            config.update(code_to_g=1, code_to_cm=C, code_to_s=1)
        if case == "gaussian_pulse":
            if not args.generator:
                raise ValueError("Gaussian application run requires --generator")
            reference = folder/"reference.bin"
            command = [str(args.generator.resolve()), "--output", str(reference), "--cells", str(8*2**level),
                       "--length", "2", "--c", "1", "--chi", str(config["rad_test_chi"]),
                       "--width", str(config["rad_test_width"]), "--background", str(config["rad_test_background"]),
                       "--amplitude", str(config["rad_test_amplitude"]), "--time", str(config["stop_time"])]
            with (folder/"reference.log").open("w") as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                               env=limited_environment(),timeout=args.timeout)
            config["rad_reference"] = str(reference)
        (folder/"run.ini").write_text("".join("%s=%s\n" % item for item in config.items()))
        command = [str(args.executable.resolve()), "--config_file="+str(folder/"run.ini"),
                   "--hpx:threads="+str(args.threads)]
        report = dict(case=case,finest_level=level,scope="Bounded application AMR smoke; not full-time accuracy validation",
                      command=command,requested_steps=args.steps,threshold_proposal_status=row["status"],status="running")
        write_json(folder/"validation.json",report)
        started = time.monotonic()
        try:
            with (folder/"run.log").open("w") as log:
                report["process_resources"] = run_bounded(command,folder,log,args.timeout,args.memory_limit_gib)
            report["histograms"] = read_histogram(folder/"gradient-amr.csv",level,args.allow_all_fine_through)
            report["conservation"] = check_ledger(folder/"radiation-conservation.csv")
            report["status"] = "application_checks_passed"
            report["all_observed_ratios_in_target"] = all(item["status"] == "ratio_in_target" for item in report["histograms"])
            report["ratio_target_or_allowed_coarse_exception"] = all(item["status"] in
                ("ratio_in_target","all_fine_coarse_exception") for item in report["histograms"])
        except Exception as error:
            report.update(status="failed",error=str(error))
            try:
                report["partial_evidence"] = partial_evidence(folder)
            except Exception as evidence_error:
                report["partial_evidence_error"] = str(evidence_error)
            if not args.keep_going:
                raise
        finally:
            report["wall_seconds"] = time.monotonic()-started
            write_json(folder/"validation.json",report)
            reports.append(report)
            write_json(output/"summary.json",dict(scope="Sequential bounded local smokes; no full-duration accuracy validation",
                       requested_cases=args.cases,requested_finest_levels=args.levels,reports=reports))
        detail = report["histograms"][-1]["status"] if "histograms" in report else report.get("error","")
        print("%s L%d: %s; %s" % (case,level,report["status"],detail),flush=True)
    if any(report["status"] == "failed" for report in reports):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    prepare_parser = commands.add_parser("prepare",help="Compile one-thread profile sampler and propose initial thresholds")
    prepare_parser.add_argument("--cxx",default="g++")
    prepare_parser.add_argument("--mode",choices=("absolute","relative"),default="absolute")
    prepare_parser.add_argument("--floor",type=float,default=1e-30)
    prepare_parser.add_argument("--buffer",type=int,choices=(0,1,2),default=0)
    prepare_parser.add_argument("--blast-density",action="store_true")
    prepare_parser.add_argument("--sample-timeout",type=float,default=180)
    run_parser = commands.add_parser("run",help="Run explicit local application smokes using a reviewed plan")
    run_parser.add_argument("--plan",type=Path,required=True)
    run_parser.add_argument("--executable",type=Path,required=True)
    run_parser.add_argument("--generator",type=Path)
    run_parser.add_argument("--threads",type=int,choices=range(1,9),default=1,
                            help="HPX workers; coordinate the shared local CPU budget before increasing")
    run_parser.add_argument("--steps",type=int,default=6)
    run_parser.add_argument("--timeout",type=float,default=300)
    run_parser.add_argument("--memory-limit-gib",type=float,default=16,
                            help="Stop a local application if sampled RSS exceeds this or half available memory")
    run_parser.add_argument("--allow-all-fine-through",type=int,choices=(1,2,3,4,5),default=3)
    run_parser.add_argument("--keep-going",action="store_true",
                            help="Record failed case/levels and continue sequentially; final exit remains nonzero")
    for subparser in (prepare_parser,run_parser):
        subparser.add_argument("--output",type=Path,required=True)
        subparser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
        subparser.add_argument("--levels",nargs="+",type=int,choices=(2,3,4,5),default=[2,3,4,5])
    args = parser.parse_args()
    if args.command == "run" and (not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0):
        parser.error("--memory-limit-gib must be finite and positive")
    if args.command == "prepare" and hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
