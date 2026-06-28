import json
import os
import re
from types import SimpleNamespace

import awkward as ak
import numpy as np
import uproot
from uproot.source.file import MemmapSource

from logger import _logger


def as_namespace(obj):
    if isinstance(obj, SimpleNamespace):
        return obj
    return SimpleNamespace(**obj)


def variation_to_fit_name(variation):
    mapping = {
        "nominal": "nominal",
        "jes_up": "jesUp",
        "jes_down": "jesDown",
        "jer_up": "jerUp",
        "jer_down": "jerDown",
        "met_up": "metUp",
        "met_down": "metDown",
    }
    return mapping[variation]


def fit_name_to_variation(fit_name):
    mapping = {
        "nominal": "nominal",
        "jesUp": "jes_up",
        "jesDown": "jes_down",
        "jerUp": "jer_up",
        "jerDown": "jer_down",
        "metUp": "met_up",
        "metDown": "met_down",
    }
    return mapping[fit_name]


def syst_to_shape_names(syst):
    if syst == "nominal":
        return ["nominal"]
    if syst in ("pu", "jes", "jer", "met", "jms", "jmr", "lhescalemuf", "lhescalemur"):
        return [syst + "Up", syst + "Down"]
    raise ValueError(f"Unsupported systematic: {syst}")


def all_shape_variations(systematics):
    out = ["nominal"]
    for syst in systematics:
        if syst != "nominal":
            out.extend(syst_to_shape_names(syst))
    return out


def clean_sample_name(path, variation):
    name = os.path.basename(path)
    if name.endswith(".root"):
        name = name[:-5]
    suffix = "_" + variation
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name


def dataset_key(sample, variation):
    return f"{sample}__{variation}"


def parse_dataset_key(key):
    sample, variation = key.rsplit("__", 1)
    return sample, variation


def mc_sample_group_map(cfg):
    return getattr(cfg, "mc_sample_groups_and_xsecs", {})


def resolve_sample_base(cfg, attr="sample_base", append_year_version=True):
    sample_base = getattr(cfg, attr)
    if sample_base is None:
        raise ValueError(f"Missing required topwsf config: {attr}")
    sample_base = sample_base.replace("$YEAR", str(cfg.year)).replace("<year>", str(cfg.year))
    nano_version = str(cfg.nano_version)
    sample_base = sample_base.replace("$NANO_VERSION", nano_version).replace("<nano_version>", nano_version)
    if append_year_version:
        suffix = f"_{cfg.year}_{nano_version}"
        if not os.path.basename(sample_base).endswith(suffix):
            sample_base += suffix
    return sample_base


def resolve_topwsf_fileset(cfg):
    sample_base = resolve_sample_base(cfg)
    mc_groups = mc_sample_group_map(cfg)
    enabled_groups = set(getattr(cfg, "enabled_sample_groups", mc_groups.keys()))
    sample_to_group = {}
    sample_to_xsec = {}
    for group, samples in mc_groups.items():
        if group not in enabled_groups:
            continue
        for sample, xsec in samples.items():
            sample_to_group[sample] = group
            sample_to_xsec[sample] = float(xsec)
    data_samples = set(getattr(cfg, "data_samples", []))

    variations = set(["nominal"])
    for syst in cfg.systematics:
        if syst in ("jes", "jer", "met"):
            variations.update([f"{syst}_up", f"{syst}_down"])

    fileset = {}
    file_metadata = {}
    missing = []
    found_nominal_samples = set()
    for variation in sorted(variations):
        vdir = os.path.join(sample_base, variation)
        if not os.path.isdir(vdir):
            missing.append(vdir)
            continue
        for path in sorted(os.path.join(vdir, f) for f in os.listdir(vdir) if f.endswith(".root")):
            sample = clean_sample_name(path, variation)
            if sample in data_samples and variation != "nominal":
                continue
            if sample not in sample_to_group and sample not in data_samples:
                continue
            if variation == "nominal":
                found_nominal_samples.add(sample)
            key = dataset_key(sample, variation)
            fileset[key] = [path]
            file_metadata[key] = {
                "sample": sample,
                "variation": variation,
                "group": "data" if sample in data_samples else sample_to_group[sample],
                "path": path,
            }
            if sample in sample_to_xsec:
                file_metadata[key]["xsec_pb"] = sample_to_xsec[sample]
    if missing:
        raise FileNotFoundError("Missing topwsf variation directories: " + ", ".join(missing))
    for sample in sorted(set(sample_to_group) - found_nominal_samples):
        _logger.info(f"[topwsf]: skip configured MC sample with no nominal input file: {sample}")
    return fileset, file_metadata


def _sum_lhe_scale_sumw(runs):
    if "LHEScaleSumw" not in runs.keys():
        return None
    arr = runs["LHEScaleSumw"].array(library="ak")
    if len(arr) == 0:
        return None
    return np.asarray(ak.sum(arr, axis=0), dtype=float)


def _lhe_scale_norms(lhe_scale_sumw, nominal_index):
    if lhe_scale_sumw is None or nominal_index >= len(lhe_scale_sumw):
        return {}, False
    nominal_sumw = float(lhe_scale_sumw[nominal_index])
    if abs(nominal_sumw) <= 1e-20:
        return {}, False
    norms = {}
    for index, sumw in enumerate(lhe_scale_sumw):
        sumw = float(sumw)
        if abs(sumw) > 1e-20:
            norms[str(index)] = nominal_sumw / sumw
    return norms, True


def compute_xsec_weights(cfg, file_metadata):
    out = {}
    skipped_samples = set()
    nominal_lhe_index = int(cfg.lhe_scale_weights["nominal"])
    for key, meta in sorted(file_metadata.items()):
        if meta["variation"] != "nominal" or meta["group"] == "data":
            continue
        sample = meta["sample"]
        if sample in skipped_samples:
            continue
        if "xsec_pb" not in meta:
            raise KeyError(f"Missing xsec for sample {sample}")
        with uproot.open(meta["path"], handler=MemmapSource) as f:
            if "Runs" not in f:
                _logger.info(f"[topwsf]: skip MC sample without Runs tree: {sample} ({meta['path']})")
                skipped_samples.add(sample)
                continue
            runs = f["Runs"]
            if "genEventSumw" not in runs.keys():
                _logger.info(f"[topwsf]: skip MC sample without Runs/genEventSumw: {sample} ({meta['path']})")
                skipped_samples.add(sample)
                continue
            sumw = float(np.sum(runs["genEventSumw"].array(library="np")))
            count = int(np.sum(runs["genEventCount"].array(library="np"))) if "genEventCount" in runs.keys() else -1
            lhe_scale_sumw = _sum_lhe_scale_sumw(runs)
        if abs(sumw) <= 1e-20:
            _logger.info(f"[topwsf]: skip MC sample with zero Runs/genEventSumw: {sample} ({meta['path']})")
            skipped_samples.add(sample)
            continue
        lhe_scale_norm, lhe_scale_norm_available = _lhe_scale_norms(lhe_scale_sumw, nominal_lhe_index)
        xsec = float(meta["xsec_pb"])
        out[sample] = {
            "group": meta["group"],
            "xsec_pb": xsec,
            "genEventSumw": sumw,
            "genEventCount": count,
            "xsecWeight": xsec * 1000.0 / sumw,
            "lheScaleSumw": [] if lhe_scale_sumw is None else [float(x) for x in lhe_scale_sumw],
            "lheScaleNorm": lhe_scale_norm,
            "lheScaleNormAvailable": lhe_scale_norm_available,
        }
    if skipped_samples:
        for key in [key for key, meta in file_metadata.items() if meta["sample"] in skipped_samples]:
            del file_metadata[key]
    return out


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, sort_keys=True)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def pt_name(pt_range):
    lo, hi = pt_range
    return f"pt{int(lo)}to{int(hi)}"


def float_token(value):
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def parse_sf_lines(path, categories):
    values = {cat: {"central": np.nan, "err_low": np.nan, "err_high": np.nan} for cat in categories}
    if not os.path.isfile(path):
        return values
    pattern = re.compile(r"(SF_[A-Za-z0-9_]+)\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:[-+]\s*)?([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)?\s*/?\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)?", re.IGNORECASE)
    with open(path) as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            name = match.group(1).replace("SF_", "")
            if name not in values:
                continue
            center = float(match.group(2))
            e1 = match.group(3)
            e2 = match.group(4)
            values[name]["central"] = center
            if e1 is not None and e2 is not None:
                values[name]["err_low"] = -abs(float(e1))
                values[name]["err_high"] = abs(float(e2))
    return values
