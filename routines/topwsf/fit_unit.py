import json
import os
import pickle
import re
import shutil
import subprocess
import html
from types import SimpleNamespace

import boost_histogram as bh
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from logger import _logger
from routines.base import ProcessingUnit, StandaloneMultiThreadedUnit
from utils.bh_tools import bh_to_uproot, fix_bh
from utils.plotting import adjust_square_hist_layout, cms_label, plt_savefig_infinite
from utils.routine_naming import routine_output_name
from utils.web_maker import WebMaker

from .common import parse_sf_lines, pt_name, write_json


TOPWSF_CMSSW_DIR = os.path.join(os.path.dirname(__file__), "cmssw")
CMSSW_WRAPPER = os.path.join(TOPWSF_CMSSW_DIR, "wrapper.sh")
CMSSW_ENV_SETUP = os.path.join(TOPWSF_CMSSW_DIR, "env_setup.sh")
TOPWSF_LAUNCH_FIT = os.path.join(TOPWSF_CMSSW_DIR, "launch_fit.sh")


def runcmd(cmd, shell=True):
    p = subprocess.Popen(cmd, shell=shell, universal_newlines=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    out, _ = p.communicate()
    return out, p.returncode


def _pt_text(pt_range):
    lo, hi = pt_range
    return f"$p_T$: [{lo:g}, {hi:g}] GeV"


def _project_mass_payload(payload, process_name, wp, region, variation, pt_range):
    key = f"{wp}__{process_name}__{variation}__{region}"
    template = payload["templates"][key]
    mass_edges = np.asarray(payload["mass_edges"], dtype=float)
    pt_edges = np.asarray(payload["pt_edges"], dtype=float)
    pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    lo, hi = pt_range
    mask = (pt_centers >= lo) & (pt_centers < hi)
    hout = bh.Histogram(bh.axis.Variable(mass_edges), storage=bh.storage.Weight())
    view = hout.view()
    values = np.asarray(template["value"], dtype=float)
    variances = np.asarray(template["variance"], dtype=float)
    view.value = np.sum(values[:, mask], axis=1)
    view.variance = np.sum(variances[:, mask], axis=1)
    return hout


def _draw_hist_step(ax, edges, values, **kwargs):
    values = np.asarray(values, dtype=float)
    ax.step(edges, np.r_[values, values[-1]], where="post", **kwargs)


def _ratio_ylim(*arrays):
    finite = np.concatenate([np.asarray(arr, dtype=float).ravel() for arr in arrays])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.5, 1.5
    return max(0.0, min(0.5, float(np.min(finite)) * 0.9)), max(1.5, float(np.max(finite)) * 1.1)


def _shape_name(process_name, syst, direction):
    if syst in ("jms", "jmr"):
        return f"{process_name}_{process_name}{syst}{direction}"
    return f"{process_name}_{syst}{direction}"


def _preview_datacard_content(inputdir, processes, systematics, lumi_unce, auto_mc_stats):
    """Build a lightweight datacard preview; the real fit card is written by cmssw/fit.py."""
    rates = {}
    obs = {}
    for region in ["pass", "fail"]:
        fpath = os.path.join(inputdir, f"inputs_{region}.root")
        with uproot.open(fpath) as f:
            obs[region] = float(np.sum(f["data_obs"].values()))
            for proc in processes:
                rates[(region, proc)] = float(np.sum(f[proc].values()))

    proc_nums = {"tp3": 4, "tp2": 3, "tp1": 6, "other": -7}
    bins = ["pass"] * len(processes) + ["fail"] * len(processes)
    proc_line = processes + processes
    proc_id_line = [str(proc_nums.get(proc, i + 1)) for i, proc in enumerate(processes)] * 2
    rate_line = [f"{rates[(region, proc)]:.8g}" for region in ["pass", "fail"] for proc in processes]

    lines = [
        "imax 2  number of channels",
        f"jmax {len(processes) - 1}  number of processes -1",
        "kmax *  number of nuisance parameters",
        "------------",
        "shapes  *  pass  inputs_pass.root  $PROCESS $PROCESS_$SYSTEMATIC",
        "shapes  *  fail  inputs_fail.root  $PROCESS $PROCESS_$SYSTEMATIC",
        "------------",
        "bin             pass fail",
        f"observation     {obs['pass']:.8g} {obs['fail']:.8g}",
        "------------",
        "bin             " + " ".join(bins),
        "process         " + " ".join(proc_line),
        "process         " + " ".join(proc_id_line),
        "rate            " + " ".join(rate_line),
        "------------",
        "lumi_13TeV      lnN      " + " ".join([f"{lumi_unce:.3f}"] * (2 * len(processes))),
    ]
    for proc in processes:
        if proc == "other":
            continue
        mask = ["1.05" if p == proc else "-" for p in proc_line]
        lines.append(f"{proc}_xsec".ljust(16) + "lnN      " + " ".join(mask))
    if "other" in processes:
        mask = ["5.00" if p == "other" else "-" for p in proc_line]
        lines.append("other_xsec".ljust(16) + "lnU      " + " ".join(mask))
    lines.append("")

    for syst in systematics:
        if syst == "nominal":
            continue
        if syst in ("jms", "jmr"):
            for proc in processes:
                mask = ["1" if p == proc else "-" for p in proc_line]
                lines.append(f"{proc}{syst}".ljust(16) + "shapeU   " + " ".join(mask))
        else:
            lines.append(syst.ljust(16) + "shape    " + " ".join(["1"] * (2 * len(processes))))
    lines.append("")
    for proc in ("tp3", "tp2", "tp1"):
        if proc in processes:
            lines.append(f"norm_top       rateParam    pass    {proc}      1   [0.,10.]")
            lines.append(f"norm_top       rateParam    fail    {proc}      1   [0.,10.]")
    lines.append("")
    lines.append(f"* autoMCStats {auto_mc_stats}")
    return "\n".join(lines) + "\n"


def _plot_shape_syst(inputdir, webdir, wp, pt_name_label, pt_text, syst, args):
    processes = args.plot_processes
    for region in ["pass", "fail"]:
        with uproot.open(os.path.join(inputdir, f"inputs_{region}.root")) as f:
            edges = f["data_obs"].axis().edges()
            nom = np.sum([f[p].values() for p in processes], axis=0)
            up_pieces = []
            down_pieces = []
            for p in processes:
                up_name = f"{p}_{p}{syst}Up" if syst in ("jms", "jmr") else f"{p}_{syst}Up"
                down_name = f"{p}_{p}{syst}Down" if syst in ("jms", "jmr") else f"{p}_{syst}Down"
                up_pieces.append(f[up_name].values() if up_name in f else f[p].values())
                down_pieces.append(f[down_name].values() if down_name in f else f[p].values())
            up = np.sum(up_pieces, axis=0)
            down = np.sum(down_pieces, axis=0)
        fig = plt.figure(figsize=(10, 10))
        gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        cms_label(ax, args.year, args.lumi)
        _draw_hist_step(ax, edges, nom, label="nominal", color="black")
        _draw_hist_step(ax, edges, up, label=f"{syst}Up", color="#e42536", linestyle="--")
        _draw_hist_step(ax, edges, down, label=f"{syst}Down", color="#5790fc", linestyle=":")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, max(np.max(nom), np.max(up), np.max(down), 1.0) * 1.4)
        ax.set_xticklabels([])
        ax.set_ylabel("MC events / bin")
        ax.text(0.03, 0.92, f"{args.tagger_label} ({wp})", transform=ax.transAxes, fontweight="bold", fontsize=18)
        ax.text(0.03, 0.86, f"{pt_text}, {region}, Syst: {syst}", transform=ax.transAxes, fontsize=16)
        ax.legend()

        axr = fig.add_subplot(gs[1])
        nom_clip = np.maximum(nom, 1e-20)
        ratio_up = up / nom_clip
        ratio_down = down / nom_clip
        _draw_hist_step(axr, edges, np.ones_like(nom), color="black")
        _draw_hist_step(axr, edges, ratio_up, color="#e42536", linestyle="--")
        _draw_hist_step(axr, edges, ratio_down, color="#5790fc", linestyle=":")
        ymin, ymax = _ratio_ylim(ratio_up, ratio_down)
        axr.set_xlim(edges[0], edges[-1])
        axr.set_ylim(ymin, ymax)
        axr.set_xlabel(args.xlabel)
        axr.set_ylabel("Var. / Nom.")
        axr.plot([edges[0], edges[-1]], [1.0, 1.0], color="black", linewidth=1)
        axr.plot([edges[0], edges[-1]], [0.5, 0.5], color="black", linestyle=":", linewidth=1)
        axr.plot([edges[0], edges[-1]], [1.5, 1.5], color="black", linestyle=":", linewidth=1)
        fname = os.path.join(webdir, f"shape_{syst}_{wp}_{pt_name_label}_{region}")
        adjust_square_hist_layout(fig)
        fig.savefig(fname + ".png")
        plt_savefig_infinite(fname + ".pdf")
        plt.close(fig)


def _plot_shape_syst_components(inputdir, webdir, wp, pt_name_label, pt_text, syst, args):
    processes = [proc for proc in args.plot_processes if proc in args.process_colors]
    colors = args.process_colors
    for region in ["pass", "fail"]:
        hists = {}
        with uproot.open(os.path.join(inputdir, f"inputs_{region}.root")) as f:
            edges = f["data_obs"].axis().edges()
            for proc in processes:
                up_name = f"{proc}_{proc}{syst}Up" if syst in ("jms", "jmr") else f"{proc}_{syst}Up"
                down_name = f"{proc}_{proc}{syst}Down" if syst in ("jms", "jmr") else f"{proc}_{syst}Down"
                hists[proc] = {
                    "nominal": f[proc].values(),
                    "up": f[up_name].values() if up_name in f else f[proc].values(),
                    "down": f[down_name].values() if down_name in f else f[proc].values(),
                }

        fig = plt.figure(figsize=(10, 10))
        gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        cms_label(ax, args.year, args.lumi)
        ymax = 1.0
        ratio_arrays = []
        for proc in processes:
            color = colors.get(proc, "black")
            nominal = hists[proc]["nominal"]
            up = hists[proc]["up"]
            down = hists[proc]["down"]
            ymax = max(ymax, float(np.max(nominal)), float(np.max(up)), float(np.max(down)))
            _draw_hist_step(ax, edges, nominal, color=color)
            _draw_hist_step(ax, edges, up, color=color, linestyle="--")
            _draw_hist_step(ax, edges, down, color=color, linestyle=":")
            nom_clip = np.maximum(nominal, 1e-20)
            ratio_arrays.extend([up / nom_clip, down / nom_clip])
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, ymax * 1.45)
        ax.set_xticklabels([])
        ax.set_ylabel("MC events / bin")
        ax.text(0.03, 0.92, f"{args.tagger_label} ({wp})", transform=ax.transAxes, fontweight="bold", fontsize=18)
        ax.text(0.03, 0.86, f"{pt_text}, {region}, Syst: {syst}", transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.80, "Processes: " + ", ".join(processes), transform=ax.transAxes, fontweight="bold", fontsize=15)
        process_handles = [
            mpl.lines.Line2D([0], [0], color=colors.get(proc, "black"), lw=2, label=args.process_labels.get(proc, proc))
            for proc in processes
        ]
        style_handles = [
            mpl.lines.Line2D([0], [0], color="black", lw=2, linestyle="-", label="nominal"),
            mpl.lines.Line2D([0], [0], color="black", lw=2, linestyle="--", label=f"{syst}Up"),
            mpl.lines.Line2D([0], [0], color="black", lw=2, linestyle=":", label=f"{syst}Down"),
        ]
        ax.legend(handles=process_handles + style_handles, loc="upper right", fontsize=12, ncol=2)

        axr = fig.add_subplot(gs[1])
        _draw_hist_step(axr, edges, np.ones_like(next(iter(hists.values()))["nominal"]), color="black")
        for proc in processes:
            color = colors.get(proc, "black")
            nominal = hists[proc]["nominal"]
            nom_clip = np.maximum(nominal, 1e-20)
            _draw_hist_step(axr, edges, hists[proc]["up"] / nom_clip, color=color, linestyle="--")
            _draw_hist_step(axr, edges, hists[proc]["down"] / nom_clip, color=color, linestyle=":")
        ymin, ymax = _ratio_ylim(*ratio_arrays)
        axr.set_xlim(edges[0], edges[-1])
        axr.set_ylim(ymin, ymax)
        axr.set_xlabel(args.xlabel)
        axr.set_ylabel("Var. / Nom.")
        axr.plot([edges[0], edges[-1]], [1.0, 1.0], color="black", linewidth=1)
        axr.plot([edges[0], edges[-1]], [0.5, 0.5], color="black", linestyle=":", linewidth=1)
        axr.plot([edges[0], edges[-1]], [1.5, 1.5], color="black", linestyle=":", linewidth=1)
        fname = os.path.join(webdir, f"shapeproc_{syst}_{wp}_{pt_name_label}_{region}")
        adjust_square_hist_layout(fig)
        fig.savefig(fname + ".png")
        plt_savefig_infinite(fname + ".pdf")
        plt.close(fig)


def _write_one_datacard_task(arg):
    args, templates_pickle, wp, pt_range = arg
    ptlabel = pt_name(pt_range)
    outdir = os.path.join(args.outputdir, "cards", wp, ptlabel)
    os.makedirs(outdir, exist_ok=True)

    with open(templates_pickle, "rb") as f:
        payload = pickle.load(f)
    for region in ["pass", "fail"]:
        with uproot.recreate(os.path.join(outdir, f"inputs_{region}.root")) as fw:
            data_hist = _project_mass_payload(payload, "data_obs", wp, region, "nominal", pt_range)
            fw["data_obs"] = bh_to_uproot(data_hist)
            for proc in args.processes:
                nominal = fix_bh(_project_mass_payload(payload, proc, wp, region, "nominal", pt_range))
                fw[proc] = bh_to_uproot(nominal)
                for syst in args.systematics:
                    if syst == "nominal":
                        continue
                    for direction in ("Up", "Down"):
                        variation = syst + direction
                        try:
                            hsyst = fix_bh(_project_mass_payload(payload, proc, wp, region, variation, pt_range))
                        except Exception:
                            hsyst = nominal.copy()
                        fw[_shape_name(proc, syst, direction)] = bh_to_uproot(hsyst)

    preview_card = os.path.join(outdir, "datacard_preview.txt")
    with open(preview_card, "w") as f:
        f.write(_preview_datacard_content(outdir, args.processes, args.systematics, args.lumi_unce, args.auto_mc_stats))

    if getattr(args, "make_plots", False):
        plot_args = SimpleNamespace(
            colors=args.plot_colors,
            plot_processes=args.plot_processes,
            process_colors=dict(zip(args.plot_processes, args.plot_colors)),
            process_labels=args.process_labels,
            xlabel=args.xlabel,
            year=args.year,
            lumi=args.lumi,
            tagger_label=args.tagger_label,
        )
        pt_text = _pt_text(pt_range)
        for syst in args.systematics:
            if syst == "nominal":
                continue
            _plot_shape_syst(outdir, args.webdir, wp, ptlabel, pt_text, syst, plot_args)
            _plot_shape_syst_components(outdir, args.webdir, wp, ptlabel, pt_text, syst, plot_args)

    return {
        "wp": wp,
        "pt_name": ptlabel,
        "pt_range": list(pt_range),
        "inputdir": outdir,
        "datacard_preview": preview_card,
    }


def _graph_values(obj):
    try:
        vals = obj.values(1)
        errh = obj.errors("high")[1]
        errl = obj.errors("low")[1]
        return vals, errl, errh
    except Exception:
        vals = obj.values()
        err = np.sqrt(np.maximum(vals, 0.0))
        return vals, err, err


def _post_step_values(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    return np.r_[values, values[-1]]


def _ratio_with_uncertainty(num, den, num_err, den_err):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    num_err = np.asarray(num_err, dtype=float)
    den_err = np.asarray(den_err, dtype=float)
    num_clip = np.maximum(np.abs(num), 1e-20)
    den_abs = np.abs(den)
    finite_den = den_abs[np.isfinite(den_abs)]
    min_den = max(1e-9, 1e-3 * float(np.max(finite_den))) if finite_den.size else 1e-9
    den_rel_err = np.full_like(den_abs, np.inf, dtype=float)
    num_rel_err = np.full_like(num_clip, np.inf, dtype=float)
    den_rel_err[den_abs > 0] = den_err[den_abs > 0] / den_abs[den_abs > 0]
    num_rel_err[num_clip > 0] = num_err[num_clip > 0] / num_clip[num_clip > 0]
    valid = (den_abs > min_den) & (den_rel_err < 1.0) & (num_rel_err < 1.0)
    ratio = np.full_like(num, np.nan, dtype=float)
    ratio_err = np.full_like(num, np.nan, dtype=float)
    ratio[valid] = num[valid] / den[valid]
    ratio_err[valid] = np.abs(ratio[valid]) * np.sqrt((num_err[valid] / num_clip[valid]) ** 2 + (den_err[valid] / den_abs[valid]) ** 2)
    return ratio, ratio_err


def _bin_centers(edges):
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _legend_errorbar(ax, color, label, linewidth):
    return ax.errorbar(
        [np.nan],
        [np.nan],
        yerr=[1.0],
        fmt="_",
        markersize=8,
        color=color,
        linestyle="none",
        capsize=0,
        elinewidth=linewidth,
        label=label,
    )


def _plot_fit_shapes(inputdir, workdir, args):
    fit_path = os.path.join(workdir, "fitDiagnosticsTest.root")
    if not os.path.isfile(fit_path):
        return []
    edges = uproot.open(os.path.join(inputdir, "inputs_pass.root"))["data_obs"].axis().edges()
    fit = uproot.open(fit_path)
    made = []
    for rootdir, label in [("shapes_prefit", "prefit"), ("shapes_fit_s", "postfit")]:
        if rootdir not in fit:
            continue
        for region in ["pass", "fail"]:
            if f"{rootdir}/{region}" not in fit:
                continue
            fig = plt.figure(figsize=(10, 10))
            gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax = fig.add_subplot(gs[0])
            cms_label(ax, args.year, args.lumi)
            values_mc = []
            variances_mc = []
            for proc in args.plot_processes:
                hproc = fit[f"{rootdir}/{region}/{proc}"]
                values_mc.append(hproc.values())
                variances_mc.append(hproc.variances())
            total = fit[f"{rootdir}/{region}/total"].values()
            total_err = np.sqrt(np.maximum(fit[f"{rootdir}/{region}/total"].variances(), 0.0))
            data, data_errl, data_errh = _graph_values(fit[f"shapes_prefit/{region}/data"])
            hep.histplot(
                values_mc,
                bins=edges,
                label=[args.process_labels.get(p, p) for p in args.plot_processes],
                histtype="fill",
                color=args.colors,
                edgecolor="k",
                linewidth=1,
                stack=True,
            )
            ax.fill_between(edges, _post_step_values(total - total_err), _post_step_values(total + total_err), step="post", hatch="\\\\", edgecolor="dimgrey", facecolor="none", linewidth=0, label="Total unc.")
            hep.histplot(data, yerr=(data_errl, data_errh), bins=edges, histtype="errorbar", color="k", label="Data")
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(0, max(np.max(data), np.max(total), 1.0) * 1.8)
            ax.set_ylabel("Events / bin")
            ax.set_xticklabels([])
            ax.text(0.03, 0.92, f"{args.tagger_label} ({args.wp})", transform=ax.transAxes, fontweight="bold", fontsize=18)
            ax.text(0.03, 0.86, f"{args.pt_name}, {region}, {label}", transform=ax.transAxes, fontsize=16)
            ax.legend()

            axr = fig.add_subplot(gs[1])
            total_clip = np.maximum(total, 1e-20)
            axr.fill_between(edges, _post_step_values((total - total_err) / total_clip), _post_step_values((total + total_err) / total_clip), step="post", hatch="\\\\", edgecolor="dimgrey", facecolor="none", linewidth=0)
            hep.histplot(data / total_clip, yerr=(data_errl / total_clip, data_errh / total_clip), bins=edges, histtype="errorbar", color="k")
            axr.plot([edges[0], edges[-1]], [1, 1], color="black")
            axr.set_xlim(edges[0], edges[-1])
            axr.set_ylim(0.001, 1.999)
            axr.set_xlabel(args.xlabel)
            axr.set_ylabel("Data / MC")
            fname = f"stack_{label}_{region}.png"
            adjust_square_hist_layout(fig)
            fig.savefig(os.path.join(workdir, fname))
            plt_savefig_infinite(os.path.join(workdir, fname.replace(".png", ".pdf")))
            plt.close(fig)
            made.append(fname)
    return made


def _plot_mc_prepost_shapes(inputdir, workdir, args):
    fit_path = os.path.join(workdir, "fitDiagnosticsTest.root")
    if not os.path.isfile(fit_path):
        return []
    edges = uproot.open(os.path.join(inputdir, "inputs_pass.root"))["data_obs"].axis().edges()
    fit = uproot.open(fit_path)
    made = []
    component_processes = list(args.plot_processes)
    process_colors = dict(zip(args.plot_processes, args.colors))
    centers = _bin_centers(edges)
    for region in ["pass", "fail"]:
        if f"shapes_prefit/{region}" not in fit or f"shapes_fit_s/{region}" not in fit:
            continue
        fig = plt.figure(figsize=(10, 10))
        gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        cms_label(ax, args.year, args.lumi)

        ratio_arrays = []
        ymax = 1.0
        handles = []
        for proc in component_processes:
            hpre = fit[f"shapes_prefit/{region}/{proc}"]
            hpost = fit[f"shapes_fit_s/{region}/{proc}"]
            pre = np.asarray(hpre.values(), dtype=float)
            post = np.asarray(hpost.values(), dtype=float)
            pre_err = np.sqrt(np.maximum(np.asarray(hpre.variances(), dtype=float), 0.0))
            post_err = np.sqrt(np.maximum(np.asarray(hpost.variances(), dtype=float), 0.0))
            color = process_colors.get(proc, "grey")
            label = args.process_labels.get(proc, proc)
            pre_lw = 1.8
            post_lw = 2.2
            ax.step(edges, _post_step_values(pre), where="post", color=color, linestyle="--", linewidth=pre_lw)
            ax.step(edges, _post_step_values(post), where="post", color=color, linestyle="-", linewidth=post_lw)
            ax.errorbar(centers, pre, yerr=pre_err, fmt="none", color=color, linestyle="none", capsize=0, elinewidth=pre_lw)
            ax.errorbar(centers, post, yerr=post_err, fmt="none", color=color, linestyle="none", capsize=0, elinewidth=post_lw)
            handles.extend([
                _legend_errorbar(ax, color, f"{label} pre-fit", pre_lw),
                _legend_errorbar(ax, color, f"{label} post-fit", post_lw),
            ])
            ymax = max(ymax, float(np.max(pre)) if pre.size else 0.0, float(np.max(post)) if post.size else 0.0)
            ratio, ratio_err = _ratio_with_uncertainty(post, pre, post_err, pre_err)
            ratio_arrays.extend([ratio - ratio_err, ratio + ratio_err])

        hpre_total = fit[f"shapes_prefit/{region}/total"]
        hpost_total = fit[f"shapes_fit_s/{region}/total"]
        pre_total = np.asarray(hpre_total.values(), dtype=float)
        post_total = np.asarray(hpost_total.values(), dtype=float)
        pre_total_err = np.sqrt(np.maximum(np.asarray(hpre_total.variances(), dtype=float), 0.0))
        post_total_err = np.sqrt(np.maximum(np.asarray(hpost_total.variances(), dtype=float), 0.0))
        pre_total_lw = 2.0
        post_total_lw = 2.6
        ax.step(edges, _post_step_values(pre_total), where="post", color="black", linestyle="--", linewidth=pre_total_lw)
        ax.step(edges, _post_step_values(post_total), where="post", color="black", linestyle="-", linewidth=post_total_lw)
        ax.errorbar(centers, pre_total, yerr=pre_total_err, fmt="none", color="black", linestyle="none", capsize=0, elinewidth=pre_total_lw)
        ax.errorbar(centers, post_total, yerr=post_total_err, fmt="none", color="black", linestyle="none", capsize=0, elinewidth=post_total_lw)
        handles.extend([
            _legend_errorbar(ax, "black", "Total MC pre-fit", pre_total_lw),
            _legend_errorbar(ax, "black", "Total MC post-fit", post_total_lw),
        ])
        ratio_total, ratio_total_err = _ratio_with_uncertainty(post_total, pre_total, post_total_err, pre_total_err)
        ratio_arrays.extend([ratio_total - ratio_total_err, ratio_total + ratio_total_err])
        ymax = max(ymax, float(np.max(pre_total)) if pre_total.size else 0.0, float(np.max(post_total)) if post_total.size else 0.0)

        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0.0, ymax * 1.55)
        ax.set_xticklabels([])
        ax.set_ylabel("MC events / bin")
        ax.text(0.03, 0.92, f"{args.tagger_label} ({args.wp})", transform=ax.transAxes, fontweight="bold", fontsize=18)
        ax.text(0.03, 0.86, f"{args.pt_name}, {region}, pre/post MC", transform=ax.transAxes, fontsize=16)
        ax.legend(handles=handles, ncol=2, fontsize=11, loc="upper right")

        axr = fig.add_subplot(gs[1])
        for proc in component_processes:
            hpre = fit[f"shapes_prefit/{region}/{proc}"]
            hpost = fit[f"shapes_fit_s/{region}/{proc}"]
            pre = np.asarray(hpre.values(), dtype=float)
            post = np.asarray(hpost.values(), dtype=float)
            pre_err = np.sqrt(np.maximum(np.asarray(hpre.variances(), dtype=float), 0.0))
            post_err = np.sqrt(np.maximum(np.asarray(hpost.variances(), dtype=float), 0.0))
            ratio, ratio_err = _ratio_with_uncertainty(post, pre, post_err, pre_err)
            color = process_colors.get(proc, "grey")
            axr.step(edges, _post_step_values(ratio), where="post", color=process_colors.get(proc, "grey"), linestyle="-", linewidth=1.8)
            axr.errorbar(centers, ratio, yerr=ratio_err, fmt="none", color=color, linestyle="none", capsize=0, elinewidth=1.8)
        axr.step(edges, _post_step_values(ratio_total), where="post", color="black", linestyle="-", linewidth=2.2)
        axr.errorbar(centers, ratio_total, yerr=ratio_total_err, fmt="none", color="black", linestyle="none", capsize=0, elinewidth=2.2)
        finite = np.concatenate([arr[np.isfinite(arr)] for arr in ratio_arrays if arr.size])
        if finite.size:
            ymin = max(0.0, min(0.6, float(np.min(finite)) * 0.9))
            ymax_ratio = max(1.4, float(np.max(finite)) * 1.1)
        else:
            ymin, ymax_ratio = 0.6, 1.4
        axr.set_xlim(edges[0], edges[-1])
        axr.set_ylim(ymin, ymax_ratio)
        axr.set_xlabel(args.xlabel)
        axr.set_ylabel("Post / Pre")
        axr.plot([edges[0], edges[-1]], [1.0, 1.0], color="black", linewidth=1)
        fname = f"mc_prepost_{region}.png"
        adjust_square_hist_layout(fig)
        fig.savefig(os.path.join(workdir, fname))
        plt_savefig_infinite(os.path.join(workdir, fname.replace(".png", ".pdf")))
        plt.close(fig)
        made.append(fname)
    return made


def _make_summary_plot(results, categories, outputdir, args):
    labels = [f"{r['wp']}\n{r['pt_range'][0]}-{r['pt_range'][1]}" for r in results]
    x = np.arange(len(labels))
    for cat in categories:
        centers = np.array([r["sf"][cat]["central"] for r in results])
        errl = np.array([r["sf"][cat]["err_low"] for r in results])
        errh = np.array([r["sf"][cat]["err_high"] for r in results])
        fig, ax = plt.subplots(figsize=(20, 10))
        cms_label(ax, args.year, args.lumi)
        ax.errorbar(x, centers, yerr=[-errl, errh], marker="o", linestyle="none", color="black")
        ax.axhline(1.0, color="grey", linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_ylim(0.0, 2.0)
        ax.set_ylabel(f"SF ({cat})")
        ax.text(0.03, 0.92, args.tagger_label, transform=ax.transAxes, fontweight="bold", fontsize=16, ha="left", va="top")
        fig.subplots_adjust(left=0.14, right=0.96, bottom=0.18, top=0.84)
        fig.savefig(os.path.join(outputdir, f"sf_summary_{cat}.png"))
        plt_savefig_infinite(os.path.join(outputdir, f"sf_summary_{cat}.pdf"))
        plt.close(fig)


def _ordered_fit_categories(categories, tagger_type):
    categories = list(categories)
    priority = {"w": "tp2", "top": "tp3"}.get(str(tagger_type).lower())
    if priority not in categories:
        return categories
    return [priority] + [cat for cat in categories if cat != priority]


def _fit_point_anchor(result):
    token = f"{result['wp']}-{result['pt_name']}"
    return "fit-" + re.sub(r"[^A-Za-z0-9_-]+", "-", token)


def _fit_task(arg):
    task, fit_args = arg
    workdir = os.path.join(fit_args.outputdir, task["wp"], task["pt_name"])
    os.makedirs(workdir, exist_ok=True)
    status = 0
    ext = ""
    if fit_args.run_impact:
        ext += " --run-impact"
    ext += f" --impact-parallel={fit_args.impact_parallel}"
    cmd = (
        f"bash {CMSSW_WRAPPER} {TOPWSF_LAUNCH_FIT} {task['inputdir']} {workdir} "
        f"--categories={','.join(fit_args.poi_categories)} "
        f"--processes={','.join(fit_args.processes)} "
        f"--systematics={','.join(fit_args.systematics)} "
        f"--year={fit_args.year} "
        f"--lumi-uncertainty={fit_args.lumi_uncertainty} "
        f"--auto-mc-stats={fit_args.auto_mc_stats}{ext}"
    )
    if not fit_args.skip_fit:
        out, ret = runcmd(cmd)
        status = ret
        with open(os.path.join(workdir, "launcher.log"), "w") as f:
            f.write(out)
        if ret != 0:
            _logger.error("topwsf fit failed: " + workdir + "\n" + "\n".join(out.splitlines()[-20:]))
    plot_args = SimpleNamespace(
        year=fit_args.year,
        lumi=fit_args.lumi,
        plot_processes=fit_args.plot_processes,
        colors=fit_args.colors,
        process_labels=fit_args.process_labels,
        xlabel=fit_args.xlabel,
        tagger_label=fit_args.tagger_label,
        wp=task["wp"],
        pt_name=task["pt_name"],
    )
    _plot_fit_shapes(task["inputdir"], workdir, plot_args)
    _plot_mc_prepost_shapes(task["inputdir"], workdir, plot_args)
    return {**task, "workdir": workdir, "status": status, "sf": parse_sf_lines(os.path.join(workdir, "fit.log"), fit_args.poi_categories)}


class TopWSFFitUnit(ProcessingUnit):
    def __init__(self, global_cfg, job_name="2_fit", job_name_step1="1_templates", fileset=None, **kwargs):
        super().__init__(job_name=job_name, fileset=fileset, processor_cls=None, **kwargs)
        self.global_cfg = global_cfg
        self.job_name_step1 = job_name_step1
        job_base = routine_output_name(self.global_cfg)
        self.outputdir = os.path.join("output", job_base, self.job_name)
        self.outputdir_step1 = os.path.join("output", job_base, self.job_name_step1)
        self.webdir = os.path.join("web", job_base, self.job_name)
        os.makedirs(self.outputdir, exist_ok=True)
        os.makedirs(self.webdir, exist_ok=True)

    def preprocess(self):
        path = os.path.join(self.outputdir_step1, "templates2d.pickle")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cannot find {path}. Please run topwsf step 1 first.")
        self.templates_pickle = path
        self.datacard_args = SimpleNamespace(
            outputdir=self.outputdir,
            processes=list(self.global_cfg.fit_processes),
            plot_processes=list(self.global_cfg.plot_process_order),
            plot_colors=list(self.global_cfg.plot_colors),
            process_labels=self.global_cfg.process_labels,
            xlabel="$m_{SD}$ [GeV]",
            year=str(self.global_cfg.year),
            lumi=self.global_cfg.lumi_dict[str(self.global_cfg.year)],
            tagger_label=getattr(self.global_cfg.tagger, "label", "Tagger"),
            systematics=["nominal"] + [s for s in self.global_cfg.systematics if s != "nominal"],
            lumi_unce=float(self.global_cfg.lumi_uncertainty[str(self.global_cfg.year)]),
            auto_mc_stats=int(self.global_cfg.fit_auto_mc_stats),
            webdir=self.webdir,
            make_plots=True,
        )
        self.fit_args = SimpleNamespace(
            outputdir=self.outputdir,
            poi_categories=list(self.global_cfg.fit_poi_categories[self.global_cfg.category]),
            auto_mc_stats=int(self.global_cfg.fit_auto_mc_stats),
            skip_fit=bool(self.global_cfg.skip_fit),
            run_impact=bool(self.global_cfg.fit_run_impact),
            impact_parallel=int(self.global_cfg.fit_impact_parallel),
            year=str(self.global_cfg.year),
            lumi=self.global_cfg.lumi_dict[str(self.global_cfg.year)],
            lumi_uncertainty=float(self.global_cfg.lumi_uncertainty[str(self.global_cfg.year)]),
            processes=list(self.global_cfg.fit_processes),
            systematics=[syst for syst in self.global_cfg.systematics if syst != "nominal"],
            plot_processes=list(self.global_cfg.plot_process_order),
            colors=list(self.global_cfg.plot_colors),
            process_labels=self.global_cfg.process_labels,
            xlabel="$m_{SD}$ [GeV]",
            tagger_label=getattr(self.global_cfg.tagger, "label", "Tagger"),
        )

    def postprocess(self):
        _logger.info("[topwsf step 2]: writing projected templates, preview datacards, and shape plots.")
        handler = StandaloneMultiThreadedUnit(workers=self.workers, use_unordered_mapping=True)
        for wp in self.global_cfg.tagger.wps:
            for pt_range in self.global_cfg.fit_pt_bins:
                handler.book((self.datacard_args, self.templates_pickle, wp, tuple(pt_range)))
        self.fit_tasks = handler.run(_write_one_datacard_task)
        wp_order = {wp: i for i, wp in enumerate(self.global_cfg.tagger.wps)}
        self.fit_tasks = sorted(self.fit_tasks, key=lambda task: (wp_order.get(task["wp"], 999), task["pt_range"][0], task["pt_range"][1]))
        write_json(os.path.join(self.outputdir, "fit_tasks.json"), self.fit_tasks)

        if not self.global_cfg.skip_fit:
            _logger.info("[topwsf step 2]: setting up CMSSW/combine environment.")
            out, ret = runcmd(f"bash {CMSSW_ENV_SETUP}")
            if ret != 0:
                _logger.exception("Error running CMSSW setup:\n\n" + out)
                raise RuntimeError("CMSSW setup failed")

        handler = StandaloneMultiThreadedUnit(workers=self.workers, use_unordered_mapping=True)
        for task in self.fit_tasks:
            handler.book((task, self.fit_args))
        self.fit_results = handler.run(_fit_task)
        wp_order = {wp: i for i, wp in enumerate(self.global_cfg.tagger.wps)}
        self.fit_results = sorted(self.fit_results, key=lambda task: (wp_order.get(task["wp"], 999), task["pt_range"][0], task["pt_range"][1]))
        write_json(os.path.join(self.outputdir, "fit_results.json"), self.fit_results)

    def make_webpage(self):
        path = os.path.join(self.outputdir, "fit_results.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            results = json.load(f)
        args = SimpleNamespace(
            year=str(self.global_cfg.year),
            lumi=self.global_cfg.lumi_dict[str(self.global_cfg.year)],
            tagger_label=getattr(self.global_cfg.tagger, "label", "Tagger"),
        )
        categories = _ordered_fit_categories(
            self.global_cfg.fit_poi_categories[self.global_cfg.category],
            getattr(self.global_cfg.tagger, "type", self.global_cfg.category),
        )
        _make_summary_plot(results, categories, self.webdir, args)

        web = WebMaker(self.job_name)
        web.add_h1("Fit Results")
        tasks_path = os.path.join(self.outputdir, "fit_tasks.json")
        if os.path.isfile(tasks_path):
            with open(tasks_path) as f:
                web.add_text("<details><summary>Fit input tasks</summary>")
                web.add_text("<pre><code>" + html.escape(f.read()) + "</code></pre>")
                web.add_text("</details>")
                web.add_text()
        for cat in categories:
            web.add_h2(f"SF ({cat})")
            web.add_figure(self.webdir, f"sf_summary_{cat}.png", f"SF summary {cat}", width=800, height=400)
            web.add_text()
            web.add_text("| WP / pT | central | uncertainty | status |")
            web.add_text("| :--- | :---: | :---: | :---: |")
            for result in results:
                sf = result["sf"][cat]
                web.add_text(f"| {result['wp']} / {result['pt_name']} | {sf['central']:.4g} | {sf['err_low']:+.4g}/{sf['err_high']:+.4g} | {result['status']} |")
            web.add_text()

        web.add_h1("Fit Diagnostics")
        web.add_h2("Directory")
        for result in results:
            label = f"{result['wp']} / {result['pt_name']}"
            web.add_text(f"- [{label}](#{_fit_point_anchor(result)})")
        web.add_text()
        for result in results:
            rel = os.path.relpath(result["workdir"], self.outputdir)
            local_rel = rel.replace(os.sep, "__")
            local_dir = os.path.join(self.webdir, local_rel)
            os.makedirs(local_dir, exist_ok=True)
            point_title = f"{result['wp']} {result['pt_name']}"
            web.add_text(f'<h2 id="{_fit_point_anchor(result)}">{html.escape(point_title)}</h2>')
            web.add_text()
            for fname in [
                "stack_prefit_pass.png",
                "stack_prefit_fail.png",
                "stack_postfit_pass.png",
                "stack_postfit_fail.png",
                "mc_prepost_pass.png",
                "mc_prepost_fail.png",
            ]:
                src = os.path.join(result["workdir"], fname)
                if os.path.isfile(src):
                    dst = os.path.join(local_dir, fname)
                    shutil.copy2(src, dst)
                    pdf = src.replace(".png", ".pdf")
                    if os.path.isfile(pdf):
                        shutil.copy2(pdf, dst.replace(".png", ".pdf"))
            web.add_text('<table style="border-collapse:collapse;width:100%;"><tr>')
            for title in ["Pre-fit", "Post-fit", "MC pre/post comparison"]:
                web.add_text(f'<th style="text-align:center;padding:4px;">{title}</th>')
            web.add_text("</tr><tr>")
            groups = [
                ["stack_prefit_pass.png", "stack_prefit_fail.png"],
                ["stack_postfit_pass.png", "stack_postfit_fail.png"],
                ["mc_prepost_pass.png", "mc_prepost_fail.png"],
            ]
            for group in groups:
                web.add_text('<td style="vertical-align:top;text-align:center;padding:4px;">')
                for fname in group:
                    rel_src = os.path.join(local_rel, fname)
                    if os.path.isfile(os.path.join(self.webdir, rel_src)):
                        web.add_text(f'<img src="{rel_src}" title="{result["wp"]} {result["pt_name"]} {fname}" alt="{result["wp"]} {result["pt_name"]} {fname}" style="width:360px;height:360px;"/>')
                web.add_text("</td>")
            web.add_text("</tr></table>")
            web.add_text()
            impact = os.path.join(result["workdir"], "impacts.pdf")
            if os.path.isfile(impact):
                shutil.copy2(impact, os.path.join(local_dir, "impacts.pdf"))
                web.add_pdf(self.webdir, os.path.join(local_rel, "impacts.pdf"), f"{result['wp']} {result['pt_name']} impacts", width=700, height=500)
                web.add_text()
            fit_log = os.path.join(result["workdir"], "fit.log")
            if os.path.isfile(fit_log):
                with open(fit_log) as f:
                    sf_lines = [line.rstrip() for line in f if "SF_" in line]
                web.add_text("```text\n" + ("\n".join(sf_lines) if sf_lines else "No SF lines found.") + "\n```")
                web.add_text()

            web.add_text("<details><summary>Shape variations</summary>")
            web.add_text()
            web.add_text("<p>Each systematic shows the total-MC variation and the process-component variation for pass and fail regions.</p>")
            for syst in self.global_cfg.systematics:
                if syst == "nominal":
                    continue
                web.add_text(f"<h3>{html.escape(str(syst))}</h3>")
                for region in ["pass", "fail"]:
                    for prefix, title in [("shape", "total MC"), ("shapeproc", "process components")]:
                        fname = f"{prefix}_{syst}_{result['wp']}_{result['pt_name']}_{region}.png"
                        path = os.path.join(self.webdir, fname)
                        if os.path.isfile(path):
                            alt = html.escape(f"{syst} {region} {title}")
                            web.add_text(f'<img src="{fname}" title="{alt}" alt="{alt}" style="width:360px;height:360px;"/>')
                web.add_text()
            web.add_text("</details>")
            web.add_text()

            diff_log = os.path.join(result["workdir"], "diffnuisances.log")
            diff_root = os.path.join(result["workdir"], "diffnuisances.root")
            if os.path.isfile(diff_log):
                shutil.copy2(diff_log, os.path.join(local_dir, "diffnuisances.log"))
                if os.path.isfile(diff_root):
                    shutil.copy2(diff_root, os.path.join(local_dir, "diffnuisances.root"))
                web.add_text("<details><summary>diffNuisances</summary>")
                web.add_text(f'<p><a href="{os.path.join(local_rel, "diffnuisances.log")}">diffnuisances.log</a>')
                if os.path.isfile(diff_root):
                    web.add_text(f' | <a href="{os.path.join(local_rel, "diffnuisances.root")}">diffnuisances.root</a>')
                web.add_text("</p>")
                with open(diff_log) as f:
                    lines = f.readlines()
                max_lines = 300
                payload = "".join(lines[:max_lines])
                if len(lines) > max_lines:
                    payload += f"\n... truncated after {max_lines} lines; open diffnuisances.log for the full output.\n"
                web.add_text("<pre><code>" + html.escape(payload) + "</code></pre>")
                web.add_text("</details>")
                web.add_text()

        web.write_to_file(self.webdir)
