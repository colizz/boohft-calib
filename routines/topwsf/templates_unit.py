import json
import os
import pickle
import html
from types import SimpleNamespace

import awkward as ak
from coffea import processor
import hist
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from logger import _logger
from routines.base import ProcessingUnit
from utils.plotting import adjust_square_hist_layout, cms_label, make_generic_mc_data_plots
from utils.routine_naming import routine_output_name
from utils.tools import eval_expr, expression_names
from utils.web_maker import WebMaker

from .common import (
    all_shape_variations,
    compute_xsec_weights,
    dataset_key,
    parse_dataset_key,
    pt_name,
    resolve_topwsf_fileset,
    variation_to_fit_name,
    write_json,
)


def _as_numpy(array):
    """Convert awkward/numpy-like arrays to a plain numpy array for histogram fill."""
    return np.asarray(ak.to_numpy(array))


def _deterministic_normal(lumi, event):
    # Build a reproducible per-event Gaussian number without relying on chunk order.
    lumi = np.asarray(lumi, dtype=np.uint64)
    event = np.asarray(event, dtype=np.uint64)
    seed1 = (lumi * np.uint64(1103515245) + event * np.uint64(12345) + np.uint64(0x9E3779B9))
    seed2 = (lumi * np.uint64(2654435761) + event * np.uint64(1013904223) + np.uint64(0x85EBCA6B))
    u1 = ((seed1 % np.uint64(1000003)).astype(float) + 0.5) / 1000003.0
    u2 = ((seed2 % np.uint64(1000033)).astype(float) + 0.5) / 1000033.0
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def _axis_edges(axis):
    """Return edges from a hist axis as a numpy array."""
    return np.asarray([axis.value(i) for i in range(axis.size + 1)], dtype=float)


def _validation_regions(global_cfg):
    """Build the named selections used by step-1 Data/MC validation plots."""
    regions = ["inclusive"]
    regions.extend(pt_name(pt_range) for pt_range in global_cfg.fit_pt_bins)
    regions.extend(f"pass_{wp}" for wp in global_cfg.tagger.wps)
    return regions


def _template2d_payload(h):
    """Convert the coffea histogram into a compact pickle-friendly payload."""
    # Keep step-1 templates in a plain pickle payload.  ROOT is only needed later
    # when step 2 projects these arrays into Combine input histograms.
    payload = {
        "mass_edges": _axis_edges(h.axes["mass"]),
        "pt_edges": _axis_edges(h.axes["pt"]),
        "axes": {
            "wp": list(h.axes["wp"]),
            "process": list(h.axes["process"]),
            "region": list(h.axes["region"]),
            "variation": list(h.axes["variation"]),
        },
        "templates": {},
    }
    for wp in h.axes["wp"]:
        for process_name in h.axes["process"]:
            for region in h.axes["region"]:
                for variation in h.axes["variation"]:
                    if process_name == "data_obs" and variation != "nominal":
                        continue
                    h2 = h[{"wp": wp, "process": process_name, "region": region, "variation": variation}].project("mass", "pt")
                    key = f"{wp}__{process_name}__{variation}__{region}"
                    payload["templates"][key] = {
                        "value": np.asarray(h2.values(flow=False), dtype=float),
                        "variance": np.asarray(h2.variances(flow=False), dtype=float),
                    }
    return payload


def _load_template2d_payload(path):
    """Load the step-1 template payload from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _payload_has_content(payload):
    """Check whether a stored payload has nonzero nominal template content."""
    # A zero-content payload usually means an old/broken result.pickle was loaded.
    # Fail fast in that case instead of silently overwriting useful step outputs.
    for key, template in payload.get("templates", {}).items():
        if "__nominal__" not in key:
            continue
        values = np.asarray(template["value"], dtype=float)
        if np.any(values[np.isfinite(values)]):
            return True
    return False


class TopWSFTemplatesCoffeaProcessor(processor.ProcessorABC):
    """Coffea processor that builds TopWSF templates and validation histograms."""

    def __init__(self, global_cfg=None, file_metadata=None, xsec_weights=None):
        self.global_cfg = global_cfg
        self.file_metadata = file_metadata
        self.xsec_weights = xsec_weights
        self.wps = global_cfg.tagger.wps
        self.shape_variations = all_shape_variations(global_cfg.systematics)
        self.mc_processes = list(global_cfg.fit_processes)
        self.processes = ["data_obs"] + self.mc_processes
        self.validation_regions = _validation_regions(global_cfg)
        self.tagger_span = getattr(global_cfg.tagger, "span", [0.0, 1.0])

        # Only preload branches needed by configured expressions and enabled
        # systematics.  The fallback coffea runner uses this list for uproot reads.
        branch_exprs = [
            global_cfg.selection,
            global_cfg.tagger.expr,
        ]
        if getattr(global_cfg, "custom_selection", None) not in (None, ""):
            branch_exprs.append(global_cfg.custom_selection)
        self.required_branches = {
            "genWeight",
            "puWeight",
            "puWeightUp",
            "puWeightDown",
            "topptWeight",
            "luminosityBlock",
            "event",
            "fj_1_sdmass",
            "fj_1_pt",
            "fj_1_eta",
            "fj_1_phi",
            "fj_1_dr_T_Wq_max",
            "fj_1_dr_T_b",
            "fj_1_T_Wq_max_pdgId",
            "fj_1_dr_W_daus",
        }
        if "lhescalemuf" in global_cfg.systematics or "lhescalemur" in global_cfg.systematics:
            self.required_branches.add("LHEScaleWeight")
        for expr in branch_exprs:
            self.required_branches.update(expression_names(expr))

        self._accumulator = {
            "templates": hist.Hist(
                hist.axis.StrCategory(self.processes, name="process"),
                hist.axis.StrCategory(list(self.wps.keys()), name="wp"),
                hist.axis.StrCategory(["pass", "fail"], name="region"),
                hist.axis.StrCategory(self.shape_variations, name="variation"),
                hist.axis.Regular(
                    int(global_cfg.template_mass_bins[0]),
                    float(global_cfg.template_mass_bins[1]),
                    float(global_cfg.template_mass_bins[2]),
                    name="mass",
                ),
                hist.axis.Regular(
                    int(global_cfg.template_pt_bins[0]),
                    float(global_cfg.template_pt_bins[1]),
                    float(global_cfg.template_pt_bins[2]),
                    name="pt",
                ),
                storage=hist.storage.Weight(),
            ),
            "cutflow": hist.Hist(
                hist.axis.StrCategory(list(file_metadata.keys()), name="dataset"),
                hist.axis.StrCategory(["all", "selected"], name="cut"),
                storage=hist.storage.Double(),
            ),
            "validation_mass": hist.Hist(
                hist.axis.StrCategory(self.processes, name="process"),
                hist.axis.StrCategory(self.validation_regions, name="selection"),
                hist.axis.Regular(
                    int(global_cfg.template_mass_bins[0]),
                    float(global_cfg.template_mass_bins[1]),
                    float(global_cfg.template_mass_bins[2]),
                    name="mass",
                ),
                storage=hist.storage.Weight(),
            ),
            "validation_pt": hist.Hist(
                hist.axis.StrCategory(self.processes, name="process"),
                hist.axis.StrCategory(self.validation_regions, name="selection"),
                hist.axis.Regular(
                    int(global_cfg.template_pt_bins[0]),
                    float(global_cfg.template_pt_bins[1]),
                    float(global_cfg.template_pt_bins[2]),
                    name="pt",
                ),
                storage=hist.storage.Weight(),
            ),
            "validation_tagger": hist.Hist(
                hist.axis.StrCategory(self.processes, name="process"),
                hist.axis.StrCategory(self.validation_regions, name="selection"),
                hist.axis.Regular(40, float(self.tagger_span[0]), float(self.tagger_span[1]), name="tagger"),
                storage=hist.storage.Weight(),
            ),
            "validation_eta": hist.Hist(
                hist.axis.StrCategory(self.processes, name="process"),
                hist.axis.StrCategory(self.validation_regions, name="selection"),
                hist.axis.Regular(50, -2.5, 2.5, name="eta"),
                storage=hist.storage.Weight(),
            ),
            "validation_phi": hist.Hist(
                hist.axis.StrCategory(self.processes, name="process"),
                hist.axis.StrCategory(self.validation_regions, name="selection"),
                hist.axis.Regular(64, -np.pi, np.pi, name="phi"),
                storage=hist.storage.Weight(),
            ),
            "data_etaphi": hist.Hist(
                hist.axis.Regular(50, -2.5, 2.5, name="eta"),
                hist.axis.Regular(64, -np.pi, np.pi, name="phi"),
                storage=hist.storage.Double(),
            ),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def _fill_cutflow(self, out, dataset, cut, count):
        """Record a dataset-level event count."""
        out["cutflow"].fill(
            dataset=np.array([dataset], dtype=object),
            cut=np.array([cut], dtype=object),
            weight=np.array([count]),
        )

    def _fill_template(self, out, process_name, wp, region, variation, mass, pt, weight):
        """Fill one mass-pT template for a process/WP/region/systematic."""
        mass = np.asarray(mass)
        if len(mass) == 0:
            return
        out["templates"].fill(
            process=np.full(len(mass), process_name, dtype=object),
            wp=np.full(len(mass), wp, dtype=object),
            region=np.full(len(mass), region, dtype=object),
            variation=np.full(len(mass), variation, dtype=object),
            mass=mass,
            pt=np.asarray(pt),
            weight=np.asarray(weight),
        )

    def _fill_validation(self, out, process_name, selection_name, mass, pt, tagger, eta, phi, weight):
        """Fill the one-dimensional validation histograms for one selection."""
        mass = np.asarray(mass)
        if len(mass) == 0:
            return
        proc = np.full(len(mass), process_name, dtype=object)
        sel = np.full(len(mass), selection_name, dtype=object)
        out["validation_mass"].fill(process=proc, selection=sel, mass=mass, weight=np.asarray(weight))
        out["validation_pt"].fill(process=proc, selection=sel, pt=np.asarray(pt), weight=np.asarray(weight))
        out["validation_tagger"].fill(process=proc, selection=sel, tagger=np.asarray(tagger), weight=np.asarray(weight))
        out["validation_eta"].fill(process=proc, selection=sel, eta=np.asarray(eta), weight=np.asarray(weight))
        out["validation_phi"].fill(process=proc, selection=sel, phi=np.asarray(phi), weight=np.asarray(weight))

    def _fill_validation_set(self, out, process_name, mass, pt, tagger, eta, phi, weight, proc_sel):
        """Fill inclusive, pT-binned, and pass-WP validation regions."""
        self._fill_validation(out, process_name, "inclusive", _as_numpy(mass[proc_sel]), _as_numpy(pt[proc_sel]), _as_numpy(tagger[proc_sel]), _as_numpy(eta[proc_sel]), _as_numpy(phi[proc_sel]), _as_numpy(weight[proc_sel]))
        for pt_range in self.global_cfg.fit_pt_bins:
            lo, hi = pt_range
            pt_sel = proc_sel & (pt >= lo) & (pt < hi)
            self._fill_validation(out, process_name, pt_name(pt_range), _as_numpy(mass[pt_sel]), _as_numpy(pt[pt_sel]), _as_numpy(tagger[pt_sel]), _as_numpy(eta[pt_sel]), _as_numpy(phi[pt_sel]), _as_numpy(weight[pt_sel]))
        for wp, (wpmin, wpmax) in self.wps.items():
            pass_sel = proc_sel & (tagger > wpmin) & (tagger <= wpmax)
            self._fill_validation(out, process_name, f"pass_{wp}", _as_numpy(mass[pass_sel]), _as_numpy(pt[pass_sel]), _as_numpy(tagger[pass_sel]), _as_numpy(eta[pass_sel]), _as_numpy(phi[pass_sel]), _as_numpy(weight[pass_sel]))

    def _base_weight(self, events, sample):
        """Build the nominal MC event weight from xsec, pileup, lumi, and top pT."""
        # xsecWeight is derived during preprocessing from Runs/genEventSumw and
        # the sample xsecs in the card, because these ntuples do not store it directly.
        lumi = self.global_cfg.lumi_dict[str(self.global_cfg.year)]
        weight = events["genWeight"] * self.xsec_weights[sample]["xsecWeight"] * events["puWeight"] * lumi
        if getattr(self.global_cfg, "apply_toppt_weight", True) and "topptWeight" in events.fields:
            weight = weight * events["topptWeight"]
        return weight

    def _lhe_weight(self, events, sample, index, nominal_index):
        """Return normalized LHEScaleWeight ratios for a requested weight index."""
        # Some samples do not carry LHEScaleWeight; keep the template nominal
        # rather than dropping those events.
        # Formula: LHEScaleWeight[index] / LHEScaleWeight[nominal] * (LHEScaleSumw[nominal] / LHEScaleSumw[index])
        if "LHEScaleWeight" not in events.fields:
            return ak.ones_like(events["genWeight"])
        try:
            den = events["LHEScaleWeight"][:, nominal_index]
            num = events["LHEScaleWeight"][:, index]
            event_ratio = num / ak.where(abs(den) > 1e-20, den, 1.0)
        except Exception:
            return ak.ones_like(events["genWeight"])
        sample_norm = self.xsec_weights.get(sample, {}).get("lheScaleNorm", {}).get(str(index), 1.0)
        return event_ratio * sample_norm

    def _process_selections(self, events, group):
        """Split MC into tp3/tp2/tp1/other fit processes."""
        # Top-like MC is split by generator matching into the three template
        # processes used by the tag-and-probe fit.  Non-top MC stays in "other".
        if group in self.global_cfg.top_process_groups:
            tp3 = (events["fj_1_dr_T_Wq_max"] < 0.8) & (events["fj_1_dr_T_b"] < 0.8)
            tp2 = (
                ((events["fj_1_T_Wq_max_pdgId"] == 0) & (events["fj_1_dr_W_daus"] < 0.8))
                | ((events["fj_1_T_Wq_max_pdgId"] != 0) & (events["fj_1_dr_T_b"] >= 0.8) & (events["fj_1_dr_T_Wq_max"] < 0.8))
            )
            tp1 = ~(tp3 | tp2)
            return {"tp3": tp3, "tp2": tp2, "tp1": tp1}
        return {"other": ak.ones_like(events["genWeight"], dtype=bool)}

    def process(self, events):
        """Process one coffea chunk and return filled histograms."""
        out = {key: value.copy() * 0 for key, value in self.accumulator.items()}
        dataset = events.metadata["dataset"]
        sample, input_variation = parse_dataset_key(dataset)
        meta = self.file_metadata[dataset]
        group = meta["group"]
        is_data = group == "data"
        self._fill_cutflow(out, dataset, "all", len(events))

        selected = eval_expr(self.global_cfg.selection, events)
        selected = ak.values_astype(selected, bool)
        events = events[selected]
        self._fill_cutflow(out, dataset, "selected", len(events))
        if len(events) == 0:
            return out

        custom_selection = getattr(self.global_cfg, "custom_selection", None)
        if custom_selection not in (None, ""):
            custom_selected = eval_expr(custom_selection, events)
            custom_selected = ak.values_astype(custom_selected, bool)
            events = events[custom_selected]
            self._fill_cutflow(out, dataset, "custom_selected", len(events))
            if len(events) == 0:
                return out

        tagger = eval_expr(self.global_cfg.tagger.expr, events)
        mass_nom = events["fj_1_sdmass"]
        pt = events["fj_1_pt"]
        # Template phase space is applied to fit templates and validation plots,
        # while the base event selection above remains configurable in the card.
        template_phase_space = (
            (mass_nom >= float(self.global_cfg.template_mass_bins[1]))
            & (mass_nom < float(self.global_cfg.template_mass_bins[2]))
            & (pt >= float(self.global_cfg.template_pt_bins[1]))
            & (pt < float(self.global_cfg.template_pt_bins[2]))
        )

        if is_data:
            weights_by_variation = {"nominal": ak.ones_like(pt)}
            mass_by_variation = {"nominal": mass_nom}
            process_selections = {"data_obs": ak.ones_like(pt, dtype=bool)}
            if "fj_1_phi" in events.fields:
                data_eta_phi_sel = template_phase_space
                out["data_etaphi"].fill(eta=_as_numpy(events["fj_1_eta"][data_eta_phi_sel]), phi=_as_numpy(events["fj_1_phi"][data_eta_phi_sel]))
        else:
            process_selections = self._process_selections(events, group)
            input_fit_variation = variation_to_fit_name(input_variation)
            if input_fit_variation == "nominal":
                base_weight = self._base_weight(events, sample)
                weights_by_variation = {"nominal": base_weight}
                mass_by_variation = {"nominal": mass_nom}
                # Weight-only variations are derived from nominal ntuples.
                if "pu" in self.global_cfg.systematics:
                    weights_by_variation["puUp"] = base_weight * events["puWeightUp"] / ak.where(abs(events["puWeight"]) > 1e-20, events["puWeight"], 1.0)
                    weights_by_variation["puDown"] = base_weight * events["puWeightDown"] / ak.where(abs(events["puWeight"]) > 1e-20, events["puWeight"], 1.0)
                    mass_by_variation["puUp"] = mass_nom
                    mass_by_variation["puDown"] = mass_nom
                lhe_cfg = self.global_cfg.lhe_scale_weights
                if "lhescalemuf" in self.global_cfg.systematics:
                    weights_by_variation["lhescalemufUp"] = base_weight * self._lhe_weight(events, sample, lhe_cfg["muf_up"], lhe_cfg["nominal"])
                    weights_by_variation["lhescalemufDown"] = base_weight * self._lhe_weight(events, sample, lhe_cfg["muf_down"], lhe_cfg["nominal"])
                    mass_by_variation["lhescalemufUp"] = mass_nom
                    mass_by_variation["lhescalemufDown"] = mass_nom
                if "lhescalemur" in self.global_cfg.systematics:
                    weights_by_variation["lhescalemurUp"] = base_weight * self._lhe_weight(events, sample, lhe_cfg["mur_up"], lhe_cfg["nominal"])
                    weights_by_variation["lhescalemurDown"] = base_weight * self._lhe_weight(events, sample, lhe_cfg["mur_down"], lhe_cfg["nominal"])
                    mass_by_variation["lhescalemurUp"] = mass_nom
                    mass_by_variation["lhescalemurDown"] = mass_nom
                # JMS/JMR are built from the nominal mass branch so that step 1
                # still has all mass-shape templates in a single coffea pass.
                if "jms" in self.global_cfg.systematics:
                    weights_by_variation["jmsUp"] = base_weight
                    weights_by_variation["jmsDown"] = base_weight
                    mass_by_variation["jmsUp"] = mass_nom * self.global_cfg.template_jms_up
                    mass_by_variation["jmsDown"] = mass_nom * self.global_cfg.template_jms_down
                if "jmr" in self.global_cfg.systematics:
                    # JMR up is a deterministic per-event smear.  By default the
                    # down template is nominal; optionally reflect it as
                    # 2*nominal - smeared for legacy comparisons.
                    z = _deterministic_normal(_as_numpy(events["luminosityBlock"]), _as_numpy(events["event"]))
                    smeared = mass_nom * ak.Array(1.0 + self.global_cfg.template_jmr_sigma * z)
                    weights_by_variation["jmrUp"] = base_weight
                    mass_by_variation["jmrUp"] = smeared
                    weights_by_variation["jmrDown"] = base_weight
                    mass_by_variation["jmrDown"] = mass_nom
                    if self.global_cfg.template_reflect_jmr_down:
                        weights_by_variation["jmrDownSub"] = -base_weight
                        mass_by_variation["jmrDownSub"] = smeared
            else:
                # JES/JER/MET shape templates come from dedicated shifted input
                # directories and are written under the corresponding fit name.
                if input_fit_variation not in self.shape_variations:
                    return out
                base_weight = self._base_weight(events, sample)
                weights_by_variation = {input_fit_variation: base_weight}
                mass_by_variation = {input_fit_variation: mass_nom}

        if input_variation == "nominal":
            nominal_weight = weights_by_variation["nominal"]
            eta = events["fj_1_eta"]
            phi = events["fj_1_phi"]
            for process_name, proc_sel in process_selections.items():
                self._fill_validation_set(out, process_name, mass_nom, pt, tagger, eta, phi, nominal_weight, proc_sel & template_phase_space)

        for wp, (wpmin, wpmax) in self.wps.items():
            pass_sel = (tagger > wpmin) & (tagger <= wpmax)
            region_sel = {"pass": pass_sel, "fail": ~pass_sel}
            for variation, weight in weights_by_variation.items():
                write_variation = "jmrDown" if variation == "jmrDownSub" else variation
                if write_variation not in self.shape_variations:
                    continue
                mass = mass_by_variation[variation]
                for process_name, proc_sel in process_selections.items():
                    for region, reg_sel in region_sel.items():
                        sel = proc_sel & reg_sel
                        if variation == "jmrDown" and self.global_cfg.template_reflect_jmr_down:
                            # jmrDown is implemented as 2*nominal - smeared.
                            # The negative smeared component is filled as jmrDownSub.
                            self._fill_template(out, process_name, wp, region, write_variation, _as_numpy(mass[sel]), _as_numpy(pt[sel]), 2.0 * _as_numpy(weight[sel]))
                        else:
                            self._fill_template(out, process_name, wp, region, write_variation, _as_numpy(mass[sel]), _as_numpy(pt[sel]), _as_numpy(weight[sel]))

        return out

    def postprocess(self, accumulator):
        return accumulator


class TopWSFTemplatesUnit(ProcessingUnit):
    """Step 1 unit: run coffea, persist templates, and write the webpage."""

    def __init__(self, global_cfg, job_name="1_templates", fileset=None, **kwargs):
        self.global_cfg = global_cfg
        job_base = routine_output_name(self.global_cfg)
        self.outputdir = os.path.join("output", job_base, job_name)
        self.webdir = os.path.join("web", job_base, job_name)
        os.makedirs(self.outputdir, exist_ok=True)
        os.makedirs(self.webdir, exist_ok=True)
        fileset, file_metadata = resolve_topwsf_fileset(global_cfg)
        self.xsec_weights = compute_xsec_weights(global_cfg, file_metadata)
        fileset = {key: value for key, value in fileset.items() if key in file_metadata}
        self.file_metadata = file_metadata
        super().__init__(
            job_name=job_name,
            fileset=fileset,
            processor_cls=TopWSFTemplatesCoffeaProcessor,
            processor_kwargs={
                "global_cfg": global_cfg,
                "file_metadata": file_metadata,
                "xsec_weights": self.xsec_weights,
            },
            **kwargs,
        )

    def launch(self, skip_coffea=False):
        """Run step 1, with pickle-only reuse when coffea is skipped."""
        # Step 1 owns the 2D pickle payload.  When skipping coffea, only reuse
        # that pickle; do not fall back to historical ROOT intermediates.
        self._skip_coffea = skip_coffea
        if not skip_coffea:
            super().launch(skip_coffea=False)
            return

        self.preprocess()
        if self.processor_cls is not None:
            self.initalize_processor()
            payload_path = os.path.join(self.outputdir, "templates2d.pickle")
            if not os.path.isfile(payload_path):
                self.load_pickle("result")
        self.postprocess()
        self.make_webpage()

    def postprocess(self):
        """Write result pickles, compact template payloads, and JSON summaries."""
        _logger.info("[topwsf step 1]: storing coffea templates and metadata.")
        payload_path = os.path.join(self.outputdir, "templates2d.pickle")
        if getattr(self, "_skip_coffea", False):
            if os.path.isfile(payload_path):
                payload = _load_template2d_payload(payload_path)
                if _payload_has_content(payload):
                    _logger.info("[topwsf step 1]: reusing existing templates2d.pickle.")
                    return
                _logger.warning("[topwsf step 1]: existing templates2d.pickle has no nominal content.")
            if not hasattr(self, "result") or self.result is None:
                self.load_pickle("result")

        with open(os.path.join(self.outputdir, "result.pickle"), "wb") as fw:
            pickle.dump(self.result, fw)
        write_json(os.path.join(self.outputdir, "file_metadata.json"), self.file_metadata)
        write_json(os.path.join(self.outputdir, "xsec_weight.json"), self.xsec_weights)

        h = self.result["templates"]
        # Keep a compact yield summary for quick bookkeeping checks on the web
        # page.  This includes flow bins, unlike the plotted 2D payload below.
        yields = {}
        for wp in h.axes["wp"]:
            yields[wp] = {}
            for process_name in h.axes["process"]:
                yields[wp][process_name] = {}
                for region in h.axes["region"]:
                    view = h[{"wp": wp, "process": process_name, "region": region, "variation": "nominal"}]
                    yields[wp][process_name][region] = float(np.sum(view.values(flow=True)))
        write_json(os.path.join(self.outputdir, "yield_nominal.json"), yields)

        cutflow = {}
        for dataset in self.result["cutflow"].axes["dataset"]:
            cutflow[dataset] = {}
            for cut in self.result["cutflow"].axes["cut"]:
                cutflow[dataset][cut] = int(self.result["cutflow"][{"dataset": dataset, "cut": cut}])
        write_json(os.path.join(self.outputdir, "cutflow.json"), cutflow)

        # This is the canonical step-1 template artifact consumed by step 2.
        payload = _template2d_payload(h)
        if getattr(self, "_skip_coffea", False) and not _payload_has_content(payload):
            raise RuntimeError("Loaded result.pickle has no nominal template content; rerun step 1 or provide templates2d.pickle.")
        with open(payload_path, "wb") as fw:
            pickle.dump(payload, fw)

    def _plot_2d_arrays(self, values, mass_edges, pt_edges, path, title):
        """Draw one nominal 2D mass-pT template heatmap."""
        values = np.asarray(values, dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0 or not np.any(finite_values):
            return
        fig, ax = plt.subplots(figsize=(12, 10))
        # Start the color scale at white so empty mass-pT bins are visually quiet.
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "topwsf_masspt",
            ["#ffffff", "#dbeafe", "#60a5fa", "#2563eb", "#7c2d12"],
        )
        cmap.set_under("#ffffff")
        cmap.set_bad("#ffffff")
        mesh = ax.pcolormesh(
            mass_edges,
            pt_edges,
            values.T,
            shading="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=max(float(np.max(finite_values)), 1e-12),
        )
        fig.colorbar(mesh, ax=ax, label="Events")
        ax.set_xlabel("$m_{SD}$ [GeV]")
        ax.set_ylabel("$p_T$(j) [GeV]")
        year = str(self.global_cfg.year)
        lumi_dict = getattr(self.global_cfg, "lumi_dict", {})
        lumi = lumi_dict.get(year, lumi_dict.get(self.global_cfg.year, 0.0))
        cms_label(ax, year, lumi)
        ax.text(
            0.04,
            0.95,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        adjust_square_hist_layout(fig)
        fig.savefig(path)
        plt.close(fig)

    def _plot_validation_distribution(self, hist_name, axis_name, selection_name, path, xlabel, title, plot_args=None):
        """Draw a stacked Data/MC validation distribution."""
        if not hasattr(self, "result") or hist_name not in self.result:
            return
        h = self.result[hist_name]
        if selection_name not in list(h.axes["selection"]):
            return
        edges = _axis_edges(h.axes[axis_name])
        processes = [proc for proc in self.global_cfg.plot_process_order if proc in list(h.axes["process"])]
        values_mc = []
        variances_mc = []
        for proc in processes:
            hproc = h[{"process": proc, "selection": selection_name}].project(axis_name)
            values_mc.append(np.asarray(hproc.values(flow=False), dtype=float))
            variances_mc.append(np.asarray(hproc.variances(flow=False), dtype=float))
        hdata = h[{"process": "data_obs", "selection": selection_name}].project(axis_name)
        values_data = np.asarray(hdata.values(flow=False), dtype=float)
        variances_data = np.asarray(hdata.variances(flow=False), dtype=float)
        if not values_mc or (not np.any(values_data) and not np.any(np.sum(values_mc, axis=0))):
            return
        yerr_mc = np.sqrt(np.maximum(np.sum(variances_mc, axis=0), 0.0))
        yerr_data = np.sqrt(np.maximum(variances_data, 0.0))
        make_generic_mc_data_plots(
            edges,
            values_mc,
            yerr_mc,
            values_data,
            yerr_data,
            yerr_data,
            [self.global_cfg.process_labels.get(proc, proc) for proc in processes],
            [self.global_cfg.plot_colors[self.global_cfg.plot_process_order.index(proc)] for proc in processes],
            xlabel,
            "Events / bin",
            str(self.global_cfg.year),
            self.global_cfg.lumi_dict[str(self.global_cfg.year)],
            plot_text=self.global_cfg.tagger.label,
            plot_subtext=title,
            plot_args=plot_args or {},
            store_name=path[:-4] if path.endswith(".png") else path,
        )

    def _plot_data_etaphi(self, path):
        """Draw the data-only eta-phi occupancy map."""
        if not hasattr(self, "result") or "data_etaphi" not in self.result:
            return
        h = self.result["data_etaphi"]
        values = np.asarray(h.values(flow=False), dtype=float)
        if not np.any(values):
            return
        eta_edges = _axis_edges(h.axes["eta"])
        phi_edges = _axis_edges(h.axes["phi"])
        fig, ax = plt.subplots(figsize=(10, 10))
        mesh = ax.pcolormesh(eta_edges, phi_edges, values.T, shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax, label="Data events")
        ax.set_xlabel(r"$\eta$(j)")
        ax.set_ylabel(r"$\phi$(j)")
        cms_label(ax, str(self.global_cfg.year), self.global_cfg.lumi_dict[str(self.global_cfg.year)])
        ax.text(0.04, 0.95, r"Data jet $\eta$-$\phi$", transform=ax.transAxes, ha="left", va="top", fontsize=18, fontweight="bold", bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
        adjust_square_hist_layout(fig)
        fig.savefig(path)
        plt.close(fig)

    def _make_validation_plots(self):
        """Create all validation PNG files before assembling the webpage."""
        if not hasattr(self, "result"):
            return
        variables = [
            ("validation_mass", "mass", "$m_{SD}$ [GeV]", "mass"),
            ("validation_pt", "pt", "$p_T$(j) [GeV]", "pt"),
            ("validation_tagger", "tagger", self.global_cfg.tagger.label, "tagger"),
            ("validation_eta", "eta", r"$\eta$(j)", "eta"),
            ("validation_phi", "phi", r"$\phi$(j)", "phi"),
        ]
        for selection_name in _validation_regions(self.global_cfg):
            for hist_name, axis_name, xlabel, token in variables:
                title = selection_name.replace("_", " ")
                path = os.path.join(self.webdir, f"validation_{selection_name}_{token}.png")
                self._plot_validation_distribution(hist_name, axis_name, selection_name, path, xlabel, title)
                if token == "tagger":
                    log_path = os.path.join(self.webdir, f"validation_{selection_name}_{token}_log.png")
                    self._plot_validation_distribution(
                        hist_name,
                        axis_name,
                        selection_name,
                        log_path,
                        xlabel,
                        f"{title}, log scale",
                        plot_args={"ylog": True},
                    )
        self._plot_data_etaphi(os.path.join(self.webdir, "data_etaphi.png"))

    def _add_validation_figures(self, web, selection_name):
        """Add one validation selection block to the webpage."""
        labels = [
            ("mass", "mSD"),
            ("pt", "pT"),
            ("tagger", "tagger"),
            ("tagger_log", "tagger log-y"),
            ("eta", "eta"),
            ("phi", "phi"),
        ]
        for token, label in labels:
            web.add_figure(self.webdir, f"validation_{selection_name}_{token}.png", f"{selection_name} {label}", width=300, height=300)
        web.add_text()

    def _add_collapsible_json(self, web, title, payload):
        """Add a collapsible JSON block to keep the webpage readable."""
        if not isinstance(payload, str):
            payload = json.dumps(payload, indent=4, sort_keys=True)
        web.add_text(f"<details><summary>{html.escape(title)}</summary>")
        web.add_text(f"<pre><code>{html.escape(payload)}</code></pre>")
        web.add_text("</details>")
        web.add_text()

    def _add_inclusive_plot_description(self, web):
        """Document the selections used for the inclusive validation plots."""
        mass_low, mass_high = self.global_cfg.template_mass_bins[1], self.global_cfg.template_mass_bins[2]
        pt_low, pt_high = self.global_cfg.template_pt_bins[1], self.global_cfg.template_pt_bins[2]
        custom_selection = getattr(self.global_cfg, "custom_selection", None)
        custom_selection = "None" if custom_selection in (None, "") else custom_selection
        producer_url = "https://github.com/colizz/nano.cpp/blob/main/src/producers/HeavyFlavMuonSampleProducer.cpp"
        web.add_text("These inclusive plots use nominal events after the muon-channel ntuple producer selection: one tight isolated muon, corrected MET > 50 GeV, a reconstructed leptonic W, at least one medium-b-tagged AK4 jet separated from the muon, and at least one probe AK8 jet separated from the muon. The producer keeps the leading probe AK8 jet for the ntuple.\n")
        web.add_text(f"Producer reference: [HeavyFlavMuonSampleProducer.cpp]({producer_url}).\n")
        web.add_text("On top of the producer selection, step 1 applies:\n")
        web.add_text(f"1. base selection: `{self.global_cfg.selection}`\n")
        web.add_text(f"2. custom selection: `{custom_selection}`\n")
        web.add_text(f"3. template phase space: `(fj_1_sdmass >= {mass_low}) & (fj_1_sdmass < {mass_high}) & (fj_1_pt >= {pt_low}) & (fj_1_pt < {pt_high})`\n")
        web.add_text()

    def make_webpage(self):
        """Write the step-1 webpage and all plot images it references."""
        _logger.info("[topwsf step 1]: making template webpage.")
        payload_path = os.path.join(self.outputdir, "templates2d.pickle")
        if not os.path.isfile(payload_path):
            if not hasattr(self, "result") or self.result is None:
                self.load_pickle("result")
            payload = _template2d_payload(self.result["templates"])
        else:
            payload = _load_template2d_payload(payload_path)
        if not hasattr(self, "result") or self.result is None:
            result_path = os.path.join(self.outputdir, "result.pickle")
            if os.path.isfile(result_path):
                self.load_pickle("result")

        mass_edges = np.asarray(payload["mass_edges"], dtype=float)
        pt_edges = np.asarray(payload["pt_edges"], dtype=float)
        self._make_validation_plots()

        # Draw all nominal pass/fail templates first; the markdown page then
        # references these static PNGs and can be opened directly in a browser.
        for wp in payload["axes"]["wp"]:
            for process_name in payload["axes"]["process"]:
                for region in payload["axes"]["region"]:
                    key = f"{wp}__{process_name}__nominal__{region}"
                    if key not in payload["templates"]:
                        continue
                    self._plot_2d_arrays(
                        payload["templates"][key]["value"],
                        mass_edges,
                        pt_edges,
                        os.path.join(self.webdir, f"tmpl2d_{wp}_{process_name}_{region}.png"),
                        f"{wp} {process_name} {region}",
                    )

        web = WebMaker(self.job_name)
        web.add_h1("Templates")
        web.add_text()
        web.add_h2("Configuration")
        web.add_h3("Tagger and working points")
        web.add_text(f"Tagger: `{self.global_cfg.tagger.label}`.\n")
        web.add_text(f"Discriminant: `{self.global_cfg.tagger.expr}`.\n")
        for wp, (lo, hi) in self.global_cfg.tagger.wps.items():
            web.add_text(f"- `{wp}`: [{lo}, {hi}]\n")
        web.add_text()
        web.add_h3("Generator matching categories")
        web.add_text("`tp3`: top-merged category, defined by `fj_1_dr_T_Wq_max < 0.8` and `fj_1_dr_T_b < 0.8`.\n")
        web.add_text("`tp2`: W-merged category, defined by either `fj_1_T_Wq_max_pdgId == 0` and `fj_1_dr_W_daus < 0.8`, or `fj_1_T_Wq_max_pdgId != 0`, `fj_1_dr_T_b >= 0.8`, and `fj_1_dr_T_Wq_max < 0.8`.\n")
        web.add_text("`tp1`: non-merged top category, defined as top-process events that are not selected by `tp3` or `tp2`.\n")
        web.add_text("`other`: non-top-process MC events, i.e. samples outside `top_process_groups`, without generator matching split.\n")
        web.add_text()
        web.add_h2("Data/MC validation plots")
        web.add_h3("Inclusive")
        self._add_inclusive_plot_description(web)
        self._add_validation_figures(web, "inclusive")
        web.add_h3("pT-binned")
        for pt_range in self.global_cfg.fit_pt_bins:
            selection_name = pt_name(pt_range)
            web.add_text(f"`{selection_name}`")
            web.add_text()
            self._add_validation_figures(web, selection_name)
        web.add_h3("Pass regions")
        for wp in self.global_cfg.tagger.wps:
            selection_name = f"pass_{wp}"
            web.add_text(f"`{selection_name}`")
            web.add_text()
            self._add_validation_figures(web, selection_name)
        web.add_h3("Data eta-phi")
        web.add_figure(self.webdir, "data_etaphi.png", "Data eta-phi", width=460, height=460)
        web.add_text()

        web.add_h2("Nominal mass-pT templates")
        for wp in payload["axes"]["wp"]:
            web.add_h3(wp)
            for region in payload["axes"]["region"]:
                web.add_text(f"`{region}` region")
                web.add_text()
                for process_name in payload["axes"]["process"]:
                    web.add_figure(self.webdir, f"tmpl2d_{wp}_{process_name}_{region}.png", f"{wp} {process_name} {region}", width=432, height=360)
                web.add_text()
        web.add_h2("JSON summaries")
        self._add_collapsible_json(web, "xsec weights", self.xsec_weights)
        with open(os.path.join(self.outputdir, "yield_nominal.json")) as f:
            self._add_collapsible_json(web, "Nominal yields", f.read())
        web.write_to_file(self.webdir)
