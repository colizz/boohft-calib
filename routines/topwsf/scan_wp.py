#!/usr/bin/env python3

import argparse
import os
import re
import sys
from multiprocessing import get_context

import awkward as ak
import numpy as np
import uproot
from uproot.source.file import MemmapSource


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topwsf")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from launcher import load_global_cfg
from routines.topwsf.common import dataset_key, mc_sample_group_map, resolve_sample_base, topwsf_year_components
from utils.tools import eval_expr, expression_names


TARGET_BACKGROUND_EFFICIENCIES = [0.05, 0.025, 0.01, 0.005, 0.001]
STEP_SIZE = 200_000
N_WORKERS = 2
QCD_SCAN_GROUP = "qcd-mg"
QCD_SKIP_HT_BINS = 1
QCD_READ_FRACTIONS = [1.0, 1.0]
QCD_DEFAULT_READ_FRACTION = 0.1
SCAN_BASE_SELECTION = "(fj_1_eta > -2.4) & (fj_1_eta < 2.4) & (fj_1_pt >= 400.) & (fj_1_pt < 600.)"
SCAN_APPLY_MD_MASS_WINDOW = True
SCAN_MD_MASS_WINDOWS = {
    "top": (105.0, 210.0),
    "w": (65.0, 105.0),
}
PROCESS_BRANCHES = {
    "fj_1_dr_T_Wq_max",
    "fj_1_dr_T_b",
    "fj_1_T_Wq_max_pdgId",
    "fj_1_dr_W_daus",
}
WEIGHT_BRANCHES = {"genWeight", "puWeight", "topptWeight"}


def _as_numpy(array):
    return np.asarray(ak.to_numpy(array))


def _scan_selection_expr(cfg):
    selection = SCAN_BASE_SELECTION
    tagger_label = str(getattr(cfg.tagger, "label", ""))
    tagger_expr = str(getattr(cfg.tagger, "expr", ""))
    tagger_type = str(cfg.tagger.type).lower()
    if SCAN_APPLY_MD_MASS_WINDOW and ("md" in tagger_label.lower() or "md" in tagger_expr.lower()):
        if tagger_type not in SCAN_MD_MASS_WINDOWS:
            raise ValueError(f"Unsupported MD tagger.type={tagger_type!r}. Expected one of {sorted(SCAN_MD_MASS_WINDOWS)}.")
        lo, hi = SCAN_MD_MASS_WINDOWS[tagger_type]
        selection += f" & (fj_1_sdmass >= {lo}) & (fj_1_sdmass < {hi})"
    return selection


def _required_branches(cfg):
    selection_expr = _scan_selection_expr(cfg)
    exprs = [selection_expr, cfg.tagger.expr]
    branches = set(PROCESS_BRANCHES) | WEIGHT_BRANCHES
    branches.update(["fj_1_sdmass", "fj_1_pt"])
    for expr in exprs:
        branches.update(expression_names(expr))
    return sorted(branches)


def _available_branches(path):
    with uproot.open(path, handler=MemmapSource) as f:
        return set(f["Events"].keys())


def _num_events(path):
    with uproot.open(path, handler=MemmapSource) as f:
        return int(f["Events"].num_entries)


def _format_size(size):
    if size is None:
        return "missing"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _qcd_ht_sort_key(sample):
    match = re.search(r"(?:_HT|HT-)(\d+)(?:to(?:(\d+)|Inf))?[_-]", sample)
    if match is None:
        return (float("inf"), sample)
    return (int(match.group(1)), sample)


def _scan_read_fraction(read_index):
    if read_index < len(QCD_READ_FRACTIONS):
        return float(QCD_READ_FRACTIONS[read_index])
    return float(QCD_DEFAULT_READ_FRACTION)


def _resolve_scan_qcd_fileset(cfg):
    eras = topwsf_year_components(cfg.year)
    include_era_in_key = len(eras) > 1
    qcd_samples = mc_sample_group_map(cfg).get(QCD_SCAN_GROUP)
    if not qcd_samples:
        raise KeyError(f"Cannot find {QCD_SCAN_GROUP!r} in mc_sample_groups_and_xsecs")

    fileset = {}
    file_metadata = {}
    for era in eras:
        if str(era) not in cfg.lumi_dict:
            raise KeyError(f"Missing lumi_dict entry for topwsf scan input era {era}")
        sample_base = resolve_sample_base(cfg, attr="sample_scan_wp", year=era)
        nominal_dir = os.path.join(sample_base, "nominal")
        read_index = 0
        for index, sample in enumerate(sorted(qcd_samples, key=_qcd_ht_sort_key)):
            path = os.path.join(nominal_dir, f"{sample}_nominal.root")
            exists = os.path.isfile(path)
            size_bytes = os.path.getsize(path) if exists else None
            entries = _num_events(path) if exists else 0
            read = exists and index >= QCD_SKIP_HT_BINS
            read_fraction = _scan_read_fraction(read_index) if read else 0.0
            read_index += int(read)
            entry_stop = int(entries * read_fraction) if read_fraction < 1.0 else entries
            key = dataset_key(sample, "nominal") if not include_era_in_key else f"{sample}__{era}__nominal"
            weight_key = sample if not include_era_in_key else f"{sample}__{era}"
            fileset[key] = [path]
            file_metadata[key] = {
                "sample": sample,
                "era": era,
                "logical_year": str(cfg.year),
                "variation": "nominal",
                "group": QCD_SCAN_GROUP,
                "path": path,
                "xsec_pb": float(qcd_samples[sample]),
                "lumi": float(cfg.lumi_dict[str(era)]),
                "weight_key": weight_key,
                "exists": exists,
                "size_bytes": size_bytes,
                "entries": entries,
                "read": read,
                "read_fraction": read_fraction,
                "entry_start": 0,
                "entry_stop": entry_stop,
            }
    return fileset, file_metadata


def _print_qcd_input_info(cfg, file_metadata):
    print("QCD scan input files")
    print("-" * 110)
    sample_bases = [resolve_sample_base(cfg, attr="sample_scan_wp", year=era) for era in topwsf_year_components(cfg.year)]
    print(f"sample_scan_wp: {', '.join(sample_bases)}")
    print(f"Group: {QCD_SCAN_GROUP}")
    print(f"Drop first HT bins: {QCD_SKIP_HT_BINS}")
    read_fractions = ", ".join(f"{fraction:g}" for fraction in QCD_READ_FRACTIONS)
    print(f"Read fractions after drop: [{read_fractions}], later={QCD_DEFAULT_READ_FRACTION:g}")
    print()
    print(
        f"{'era':<9}  {'sample':<65}  {'xsec [pb]':>12}  {'size':>10}  {'read':>5}  "
        f"{'partition':>18}  path"
    )
    for meta in sorted(file_metadata.values(), key=lambda item: (str(item.get("era", "")), _qcd_ht_sort_key(item["sample"]))):
        if meta["exists"]:
            partition = f"{meta['entry_start']}:{meta['entry_stop']}/{meta['entries']} ({meta['read_fraction']:.3g})"
        else:
            partition = "missing"
        print(
            f"{str(meta.get('era', '')):<9}  "
            f"{meta['sample']:<65}  "
            f"{meta['xsec_pb']:12.6g}  "
            f"{_format_size(meta['size_bytes']):>10}  "
            f"{str(meta['read']):>5}  "
            f"{partition:>18}  "
            f"{meta['path']}"
        )
    print()


def _process_masks(events, group, top_process_groups):
    if group in top_process_groups:
        tp3 = (events["fj_1_dr_T_Wq_max"] < 0.8) & (events["fj_1_dr_T_b"] < 0.8)
        tp2 = (
            ((events["fj_1_T_Wq_max_pdgId"] == 0) & (events["fj_1_dr_W_daus"] < 0.8))
            | ((events["fj_1_T_Wq_max_pdgId"] != 0) & (events["fj_1_dr_T_b"] >= 0.8) & (events["fj_1_dr_T_Wq_max"] < 0.8))
        )
        tp1 = ~(tp3 | tp2)
        return {"tp3": tp3, "tp2": tp2, "tp1": tp1}
    return {"other": ak.ones_like(events["genWeight"], dtype=bool)}


def _signal_process(tagger_type):
    tagger_type = str(tagger_type).lower()
    if tagger_type == "w":
        return "tp2"
    if tagger_type == "top":
        return "tp3"
    raise ValueError(f"Unsupported tagger.type={tagger_type!r}. Expected 'w' or 'top'.")


def _selected_background(events, cfg, group, signal_process):
    selected = ak.values_astype(eval_expr(_scan_selection_expr(cfg), events), bool)
    events = events[selected]
    if len(events) == 0:
        return events, ak.Array([])

    process_masks = _process_masks(events, group, cfg.top_process_groups)
    bkg_mask = None
    for process_name, mask in process_masks.items():
        if process_name == signal_process:
            continue
        bkg_mask = mask if bkg_mask is None else (bkg_mask | mask)
    if bkg_mask is None:
        bkg_mask = ak.zeros_like(events["genWeight"], dtype=bool)
    return events, ak.values_astype(bkg_mask, bool)


def _event_weight(events, cfg, meta, xsec_weights):
    weight_key = meta.get("weight_key", meta["sample"])
    lumi = float(meta.get("lumi", cfg.lumi_dict[str(cfg.year)]))
    weight = (
        events["genWeight"]
        * float(xsec_weights[weight_key]["xsecWeight"])
        * float(xsec_weights[weight_key].get("readScale", 1.0))
        * events["puWeight"]
        * lumi
    )
    if getattr(cfg, "apply_toppt_weight", True) and "topptWeight" in events.fields:
        weight = weight * events["topptWeight"]
    return weight


def _compute_one_scan_xsec_weight(meta):
    with uproot.open(meta["path"], handler=MemmapSource) as f:
        runs = f["Runs"]
        if "genEventSumw" not in runs.keys():
            raise KeyError(f"Runs/genEventSumw is missing in {meta['path']}")
        sumw = float(np.sum(runs["genEventSumw"].array(library="np")))
    read_fraction = float(meta.get("read_fraction", 1.0))
    if read_fraction <= 0.0:
        raise ValueError(f"Invalid read_fraction={read_fraction} for {meta['path']}")
    return {
        "sample": meta["sample"],
        "era": meta.get("era"),
        "xsecWeight": float(meta["xsec_pb"]) * 1000.0 / sumw,
        "readScale": 1.0 / read_fraction,
    }


def _scan_one_file(path, meta, cfg, xsec_weights, signal_process, branches):
    available = _available_branches(path)
    optional = {"topptWeight"}
    missing = sorted(set(branches) - available - optional)
    if missing:
        raise KeyError(f"Missing branches in {path}: {', '.join(missing)}")
    read_branches = [branch for branch in branches if branch in available]

    group = meta["group"]
    scores = []
    weights = []
    entry_stop = meta.get("entry_stop")
    for events in uproot.iterate(
        f"{path}:Events",
        expressions=read_branches,
        step_size=STEP_SIZE,
        entry_stop=entry_stop,
        library="ak",
        handler=MemmapSource,
    ):
        events, bkg_mask = _selected_background(events, cfg, group, signal_process)
        if len(events) == 0 or not ak.any(bkg_mask):
            continue
        tagger = eval_expr(cfg.tagger.expr, events)
        weight = _event_weight(events, cfg, meta, xsec_weights)
        scores.append(_as_numpy(tagger[bkg_mask]).astype(float))
        weights.append(_as_numpy(weight[bkg_mask]).astype(float))

    if not scores:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(scores), np.concatenate(weights)


def _scan_one_task(task):
    key, meta, cfg, signal_process, branches = task
    label = key.rsplit("__", 1)[0]
    print(f"start {label}", flush=True)
    weight_key = meta.get("weight_key", meta["sample"])
    xsec_weights = {weight_key: _compute_one_scan_xsec_weight(meta)}
    scores, weights = _scan_one_file(meta["path"], meta, cfg, xsec_weights, signal_process, branches)
    return label, scores, weights


def _weighted_survival_thresholds(scores, weights, targets):
    if scores.size == 0:
        raise RuntimeError("No background events passed the scan selection.")
    finite = np.isfinite(scores) & np.isfinite(weights)
    scores = scores[finite]
    weights = weights[finite]
    order = np.argsort(scores)[::-1]
    scores = scores[order]
    weights = weights[order]

    starts = np.flatnonzero(np.r_[True, scores[1:] != scores[:-1]])
    grouped_scores = scores[starts]
    grouped_weights = np.add.reduceat(weights, starts)
    cumulative = np.cumsum(grouped_weights)
    total = float(np.sum(weights))
    if abs(total) <= 1e-20:
        raise RuntimeError("Total signed background yield is zero.")
    efficiencies = cumulative / total

    results = {}
    for target in targets:
        idx = int(np.nanargmin(np.abs(efficiencies - target)))
        results[target] = {
            "cut": float(grouped_scores[idx]),
            "achieved_efficiency": float(efficiencies[idx]),
            "passing_yield": float(cumulative[idx]),
        }
    diagnostics = {
        "total_signed_yield": total,
        "total_absolute_yield": float(np.sum(np.abs(weights))),
        "negative_weight_yield": float(np.sum(weights[weights < 0])),
        "negative_weight_abs_fraction": float(np.sum(np.abs(weights[weights < 0])) / max(np.sum(np.abs(weights)), 1e-20)),
        "non_monotonic_signed_efficiency": bool(np.any(np.diff(efficiencies) < -1e-10)),
        "n_events": int(scores.size),
    }
    return results, diagnostics


def _print_results(cfg, signal_process, scores, weights, thresholds, diagnostics):
    tagger_type = str(cfg.tagger.type).lower()
    background_processes = [proc for proc in cfg.fit_processes if proc != signal_process]
    print("TopWSF background-efficiency working point scan")
    print("=" * 55)
    print(f"Card: {cfg.config_path}")
    print(f"Year: {cfg.year}")
    print(f"Tagger label: {cfg.tagger.label}")
    print(f"Tagger type: {tagger_type}")
    print(f"Tagger expression: {cfg.tagger.expr}")
    print(f"Signal process: {signal_process}")
    print(f"Background processes: {', '.join(background_processes)}")
    print(f"Selection: ({_scan_selection_expr(cfg)})")
    print()
    print("Diagnostics")
    print("-" * 55)
    print(f"Selected background events: {diagnostics['n_events']}")
    print(f"Total signed yield: {diagnostics['total_signed_yield']:.8g}")
    print(f"Total absolute yield: {diagnostics['total_absolute_yield']:.8g}")
    print(f"Negative signed yield: {diagnostics['negative_weight_yield']:.8g}")
    print(f"Negative |weight| fraction: {diagnostics['negative_weight_abs_fraction']:.4%}")
    print(f"Signed cumulative efficiency non-monotonic: {diagnostics['non_monotonic_signed_efficiency']}")
    print()
    print("Working points")
    print("-" * 55)
    print(f"{'target bkg eff':>15}  {'cut':>14}  {'achieved':>14}  {'passing yield':>14}  wp range")
    for target in TARGET_BACKGROUND_EFFICIENCIES:
        result = thresholds[target]
        print(
            f"{target:15.5g}  "
            f"{result['cut']:14.8g}  "
            f"{result['achieved_efficiency']:14.8g}  "
            f"{result['passing_yield']:14.8g}  "
            f"[{result['cut']:.8g}, 1.0]"
        )


def main():
    parser = argparse.ArgumentParser("Scan topwsf tagger working points at fixed background efficiencies.")
    parser.add_argument("config_path", help="TopWSF YAML card, e.g. cards/topwsf/example_w_PNetMD_2018.yml")
    args = parser.parse_args()

    cfg = load_global_cfg(args.config_path, routine="topwsf")
    signal_process = _signal_process(cfg.tagger.type)
    fileset, file_metadata = _resolve_scan_qcd_fileset(cfg)
    _print_qcd_input_info(cfg, file_metadata)
    branches = _required_branches(cfg)

    tasks = []
    for key in sorted(fileset):
        meta = file_metadata[key]
        variation = meta["variation"]
        if variation != "nominal" or not meta["read"]:
            continue
        tasks.append((key, meta, cfg, signal_process, branches))
    if not tasks:
        raise RuntimeError("No readable qcd-mg files were found for the scan.")
    tasks.sort(key=lambda task: os.path.getsize(task[1]["path"]))

    all_scores = []
    all_weights = []
    with get_context("fork").Pool(processes=min(N_WORKERS, len(tasks))) as pool:
        for sample, scores, weights in pool.imap_unordered(_scan_one_task, tasks):
            if scores.size:
                all_scores.append(scores)
                all_weights.append(weights)
            print(f"processed {sample}: selected background entries = {scores.size}", flush=True)

    if not all_scores:
        raise RuntimeError("No selected background entries were found.")
    scores = np.concatenate(all_scores)
    weights = np.concatenate(all_weights)
    thresholds, diagnostics = _weighted_survival_thresholds(scores, weights, TARGET_BACKGROUND_EFFICIENCIES)
    print()
    _print_results(cfg, signal_process, scores, weights, thresholds, diagnostics)


if __name__ == "__main__":
    main()
