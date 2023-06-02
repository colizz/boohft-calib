"""
For step 3: derive the template for fit.

"""

from coffea import processor, hist
import awkward as ak
import numpy as np
import uproot
import uproot3

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cycler import cycler 
mpl.use('Agg')
mpl.rcParams['axes.prop_cycle'] = cycler(color=['blue', 'red', 'green', 'violet', 'darkorange', 'black', 'cyan', 'yellow'])

import boost_histogram as bh
import mplhep as hep
plt.style.use(hep.style.CMS)

from types import SimpleNamespace
from functools import partial
import shutil
import pickle
import json
import os

from unit import ProcessingUnit, StandaloneMultiThreadedUnit
from utils.web_maker import WebMaker
from utils.tools import lookup_pt_based_weight, parse_tagger_expr
from utils.plotting import make_generic_mc_data_plots
from utils.bh_tools import bh_to_uproot3, fix_bh, scale_bh
from utils.xgb_tools import XGBEnsemble
from logger import _logger


class TmplWriterCoffeaProcessor(processor.ProcessorABC):
    r"""The coffea processor for the coastline and template writing step"""

    def __init__(self, global_cfg=None, weight_map=None, xtagger_map=None, coastline_map=None, sfbdt_weight_map=None):
        self.global_cfg = global_cfg
        self.weight_map = weight_map
        self.xtagger_map = xtagger_map
        self.coastline_map = coastline_map
        self.sfbdt_weight_map = sfbdt_weight_map
        self.pt_edges = global_cfg.pt_edges + [100000]
        self.pt_reweight_edges = [edge[0] for edge in global_cfg.rwgt_pt_bins]
        self.wps = global_cfg.tagger.wps # wps in the dict format

        self.tagger_expr = parse_tagger_expr(global_cfg.tagger_name_replace_map, global_cfg.tagger.expr)
        self.lookup_mc_weight = partial(lookup_pt_based_weight, self.weight_map, self.pt_reweight_edges, jet_var_maxlimit=2500.)
        self.lookup_sfbdt_weight = partial(lookup_pt_based_weight, self.sfbdt_weight_map, self.pt_reweight_edges, jet_var_maxlimit=1.)
        self.untypes = ['nominal', 'fracBCLUp', 'fracBCLDown', 'puUp', 'puDown', 'l1PreFiringUp', 'l1PreFiringDown', 'jesUp', 'jesDown', 'jerUp', 'jerDown', 'psWeightIsrUp', 'psWeightIsrDown', 'psWeightFsrUp', 'psWeightFsrDown', 'sfBDTRwgtUp']
        self.write_untypes = [
            'nominal', 'puUp', 'puDown', 'l1PreFiringUp', 'l1PreFiringDown', 'jesUp', 'jesDown', 'jerUp', 'jerDown', \
            'fracBBUp', 'fracBBDown', 'fracCCUp', 'fracCCDown', 'fracLightUp', 'fracLightDown', \
            'psWeightIsrUp', 'psWeightIsrDown', 'psWeightFsrUp', 'psWeightFsrDown', \
            'sfBDTRwgtUp', 'sfBDTRwgtDown', 'fitVarRwgtUp', 'fitVarRwgtDown'
        ]
        if self.global_cfg.custom_sfbdt_path is not None:
            self.xgb = XGBEnsemble(
                [self.global_cfg.custom_sfbdt_path + '.%d' % i for i in range(self.global_cfg.custom_sfbdt_kfold)],
                ['fj_2_tau21', 'fj_2_sj1_rawmass', 'fj_2_sj2_rawmass', 'fj_2_ntracks_sv12', 'fj_2_sj1_sv1_pt', 'fj_2_sj2_sv1_pt'],
            )

        dataset = hist.Cat("dataset", "dataset")
        flv_bin = hist.Bin('flv', 'flv', [-.5, .5, 1.5, 2.5]) # three bins for flvL=0, flvB=1, flvC=2
        passwp_bin = hist.Bin('passwp', 'passwp', [-.5, .5, 1.5]) # two bins for fail=0, pass=1
        logmsv_bin_edges = [-0.8, -0.4, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.5, 3.2]
        logmsv_bin = hist.Bin('logmsv', 'logmsv', logmsv_bin_edges)
        self.incl_var_dict = { # key: (label, bin args)
            'sfbdt': ('sfBDT', (50, 0., 1.)),
            'xtagger': ('Transformed tagger', (50, 0., 1.)),
            'logmsv': (r'$log(m_{SV1,d_{xy}sig\,max}\; /GeV)$', (logmsv_bin_edges,)),
            'eta': (r'$\eta(j)$', (50, -2., 2.)),
            'pt': (r'$p_{T}(j)$', (40, 200., 1000.)),
            'mass': (r'$m_{SD}(j)$', (15, 50., 200.)),
        }

        hist_fit, hist_incl = {}, {}
        for ipt, (ptmin, ptmax) in enumerate(zip(self.pt_edges[:-1], self.pt_edges[1:])):
            coastline_bin = hist.Bin('coastline', 'coastline', coastline_map[ipt]['levels'])
            for wp in self.wps:
                for untype in self.untypes:
                    # these unce types should derive individual hist templates
                    hist_fit.update({
                        f'h_pt{ptmin}to{ptmax}_{wp}_{untype}': hist.Hist('Counts', dataset, flv_bin, passwp_bin, coastline_bin, logmsv_bin),
                    })
            # inclusive histogram (pass+fail)
            for var in self.incl_var_dict:
                _, bin_args = self.incl_var_dict[var]
                hist_incl.update({
                    f'hinc_{var}_pt{ptmin}to{ptmax}': hist.Hist('Counts', dataset, flv_bin, coastline_bin, hist.Bin('var', 'var', *bin_args)),
                })

        self._accumulator = processor.dict_accumulator({
            **hist_fit, **hist_incl,
        })

    @property
    def accumulator(self):
        return self._accumulator


    def process(self, events):
        out = self.accumulator.identity()
        dataset = events.metadata['dataset']
        is_mc = dataset != 'jetht'

        presel = ak.numexpr.evaluate('passmetfilters & (' + '|'.join(self.global_cfg.hlt_branches[self.global_cfg.year]) + ')', events)
        lumi = self.global_cfg.lumi_dict[self.global_cfg.year]

        for i in '12': # jet index
            custom_sel = ak.numexpr.evaluate(self.global_cfg.custom_selection.replace('fj_x', f'fj_{i}'), events) if self.global_cfg.custom_selection is not None else ak.ones_like(presel, dtype=bool)
            events_fj = events[(presel) & (events[f'fj_{i}_is_qualified']) & (custom_sel)]

            # calculate bin variables
            if is_mc:
                isB = events_fj[f'fj_{i}_nbhadrons'] >= 1
                isC = (~isB) & (events_fj[f'fj_{i}_nchadrons'] >= 1)
            else:
                isB = isC = ak.zeros_like(events_fj.ht)

            msv = ak.numexpr.evaluate(
                f'(fj_{i}_sj1_sv1_dxysig>fj_{i}_sj2_sv1_dxysig)*fj_{i}_sj1_sv1_masscor + (fj_{i}_sj1_sv1_dxysig<=fj_{i}_sj2_sv1_dxysig)*fj_{i}_sj2_sv1_masscor',
                events_fj
            )
            logmsv = np.log(np.maximum(msv, 1e-20))
            pt = events_fj[f'fj_{i}_pt']
            if self.global_cfg.custom_sfbdt_path is not None:
                sfbdt_inputs = {
                    'fj_2_tau21': events_fj[f'fj_{i}_tau21'],
                    'fj_2_sj1_rawmass': events_fj[f'fj_{i}_sj1_rawmass'],
                    'fj_2_sj2_rawmass': events_fj[f'fj_{i}_sj2_rawmass'],
                    'fj_2_ntracks_sv12': events_fj[f'fj_{i}_ntracks_sv12'],
                    'fj_2_sj1_sv1_pt': events_fj[f'fj_{i}_sj1_sv1_pt'],
                    'fj_2_sj2_sv1_pt': events_fj[f'fj_{i}_sj2_sv1_pt'],
                }
                sfbdt = ak.Array(self.xgb.eval(sfbdt_inputs))
            else:
                sfbdt = events_fj[f'fj_{i}_sfBDT']
            sfbdt_corr_cache = {} # prepared for JES/JER corrected version of sfBDT
            tagger = ak.numexpr.evaluate(self.tagger_expr.replace('fj_x', f'fj_{i}'), events_fj)
            xtagger = self.xtagger_map(tagger)

            # calculate weights in multiple senerios
            weight = {}
            if is_mc:
                mc_weight = self.lookup_mc_weight(f'fj{i}', pt, events_fj['ht'])
                sfbdt_weight = self.lookup_sfbdt_weight(f'fj{i}', pt, sfbdt)
                weight_base = ak.numexpr.evaluate(f'genWeight*xsecWeight*puWeight*l1PreFiringWeight*{lumi}', events_fj)
                weight['nominal'] = weight_base * mc_weight
                weight['fracBCLUp'] = weight['nominal'] * ak.numexpr.evaluate(
                    f'(fj_{i}_nbhadrons>=1) * (1.5*(fj_{i}_nbhadrons>1) + 1.5*(fj_{i}_nbhadrons<=1)) + ' + \
                    f'((fj_{i}_nbhadrons==0) & (fj_{i}_nchadrons>=1)) * (1.5*(fj_{i}_nchadrons>1) + 1.5*(fj_{i}_nchadrons<=1)) + ' + \
                    f'((fj_{i}_nbhadrons==0) & (fj_{i}_nchadrons==0)) * (1.5)', events_fj
                )
                weight['fracBCLDown'] = weight['nominal'] * ak.numexpr.evaluate(
                    f'(fj_{i}_nbhadrons>=1) * (0.5*(fj_{i}_nbhadrons>1) + 0.5*(fj_{i}_nbhadrons<=1)) + ' + \
                    f'((fj_{i}_nbhadrons==0) & (fj_{i}_nchadrons>=1)) * (0.5*(fj_{i}_nchadrons>1) + 0.5*(fj_{i}_nchadrons<=1)) + ' + \
                    f'((fj_{i}_nbhadrons==0) & (fj_{i}_nchadrons==0)) * (0.5)', events_fj
                )
                weight['puUp'] = ak.numexpr.evaluate(f'genWeight*xsecWeight*puWeightUp*l1PreFiringWeight*{lumi}', events_fj) * mc_weight
                weight['puDown'] = ak.numexpr.evaluate(f'genWeight*xsecWeight*puWeightDown*l1PreFiringWeight*{lumi}', events_fj) * mc_weight
                weight['l1PreFiringUp'] = ak.numexpr.evaluate(f'genWeight*xsecWeight*puWeight*l1PreFiringWeightUp*{lumi}', events_fj) * mc_weight
                weight['l1PreFiringDown'] = ak.numexpr.evaluate(f'genWeight*xsecWeight*puWeight*l1PreFiringWeightDown*{lumi}', events_fj) * mc_weight
                if len(events_fj) and hasattr(events_fj, 'PSWeight') and len(events_fj.PSWeight[0]) == 4:
                    # apply PSWeight only at the final stage, while still using the nominal MC reweighting map
                    weight['psWeightIsrUp'] = weight['nominal'] * events_fj.PSWeight[:,2]
                    weight['psWeightIsrDown'] = weight['nominal'] * events_fj.PSWeight[:,0]
                    weight['psWeightFsrUp'] = weight['nominal'] * events_fj.PSWeight[:,3]
                    weight['psWeightFsrDown'] = weight['nominal'] * events_fj.PSWeight[:,1]
                else:
                    weight['psWeightIsrUp'] = weight['psWeightIsrDown'] = weight['psWeightFsrUp'] = weight['psWeightFsrDown'] = weight['nominal']
                weight['sfBDTRwgtUp'] = weight['nominal'] * sfbdt_weight

            else: # fill weight=1 to data
                ones = ak.ones_like(events_fj.ht)
                for untype in self.untypes:
                    weight[untype] = ones

            passwp = {}
            for wp, (lo, hi) in self.wps.items():
                passwp[wp] = (tagger >= lo) & (tagger < hi)

            # fill histograms
            for ipt, (ptmin, ptmax) in enumerate(zip(self.pt_edges[:-1], self.pt_edges[1:])):
                # coastline fpline values should be calculated inside the pT loop as the coastline shape depends on pT
                ptsel = (pt >= ptmin) & (pt < ptmax)
                coastline_ptsel = ak.from_numpy(self.coastline_map[ipt]['fspline'](ak.to_numpy(xtagger[ptsel]), ak.to_numpy(sfbdt[ptsel])))

                # fill in fit histogram
                for wp in self.wps:
                    for untype in self.untypes:
                        # a histogram standards for a given pT range (coastline depends on this), for each WP, and for an unce type
                        if (not is_mc) or (is_mc and untype not in ['jesUp', 'jesDown', 'jerUp', 'jerDown']):
                            # histogram differs only by a specific event weight
                            out[f'h_pt{ptmin}to{ptmax}_{wp}_{untype}'].fill(
                                dataset=dataset,
                                flv=isB[ptsel] * 1 + isC[ptsel] * 2,
                                passwp=passwp[wp][ptsel],
                                coastline=coastline_ptsel,
                                logmsv=logmsv[ptsel],
                                weight=weight[untype][ptsel]
                            )
                        else:
                            # special handling for JES/JER: jet pt need to be corrected
                            suffix_to_branch = {'jesUp': '_jesUncFactorUp', 'jesDown': '_jesUncFactorDn', 'jerUp': '_jerSmearFactorUp', 'jerDown': '_jerSmearFactorDn'}
                            
                            pt_corr = events_fj[f'fj_{i}_pt'] * events_fj[f'fj_{i}{suffix_to_branch[untype]}']
                            ht_corr = events_fj[f'ht{suffix_to_branch[untype]}']
                            ptsel_corr = (pt_corr >= ptmin) & (pt_corr < ptmax)
                            logmsv_corr = np.log(np.maximum(msv * events_fj[f'fj_{i}{suffix_to_branch[untype]}'], 1e-20))
                            weight_corr = weight_base * self.lookup_mc_weight(f'fj{i}', pt_corr, ht_corr, read_suffix=f'_{untype}') # use JES/JER reweight map and corrected HT & pT variables
                            if untype not in sfbdt_corr_cache.keys():
                                assert self.global_cfg.custom_sfbdt_path is not None, \
                                    "To derive JES/JER templates, a customized sfBDT path must be specified because sfBDT will be recalculated from JES/JER corrected input"
                                sfbdt_inputs = {
                                    'fj_2_tau21': events_fj[f'fj_{i}_tau21'],
                                    'fj_2_sj1_rawmass': events_fj[f'fj_{i}_sj1_rawmass'] * events_fj[f'fj_{i}{suffix_to_branch[untype]}'],
                                    'fj_2_sj2_rawmass': events_fj[f'fj_{i}_sj2_rawmass'] * events_fj[f'fj_{i}{suffix_to_branch[untype]}'],
                                    'fj_2_ntracks_sv12': events_fj[f'fj_{i}_ntracks_sv12'],
                                    'fj_2_sj1_sv1_pt': events_fj[f'fj_{i}_sj1_sv1_pt'] * events_fj[f'fj_{i}{suffix_to_branch[untype]}'],
                                    'fj_2_sj2_sv1_pt': events_fj[f'fj_{i}_sj2_sv1_pt'] * events_fj[f'fj_{i}{suffix_to_branch[untype]}'],
                                }
                                sfbdt_corr_cache[untype] = ak.Array(self.xgb.eval(sfbdt_inputs))
                            coastline_ptsel_corr = ak.from_numpy(self.coastline_map[ipt]['fspline'](ak.to_numpy(xtagger[ptsel_corr]), ak.to_numpy(sfbdt_corr_cache[untype][ptsel_corr])))
                            out[f'h_pt{ptmin}to{ptmax}_{wp}_{untype}'].fill(
                                dataset=dataset,
                                flv=isB[ptsel_corr] * 1 + isC[ptsel_corr] * 2,
                                passwp=passwp[wp][ptsel_corr],
                                coastline=coastline_ptsel_corr,
                                logmsv=logmsv_corr[ptsel_corr],
                                weight=weight_corr[ptsel_corr]
                            )

                # fill in inclusive histogram
                for var, expr in zip(self.incl_var_dict.keys(), [
                    'sfbdt[ptsel]', 'xtagger[ptsel]', 'logmsv[ptsel]', \
                    f'events_fj.fj_{i}_eta[ptsel]', f'events_fj.fj_{i}_pt[ptsel]', f'events_fj.fj_{i}_sdmass[ptsel]',
                ]):
                    out[f'hinc_{var}_pt{ptmin}to{ptmax}'].fill(
                        dataset=dataset,
                        flv=isB[ptsel] * 1 + isC[ptsel] * 2,
                        coastline=coastline_ptsel,
                        var=eval(expr),
                        weight=weight['nominal'][ptsel]
                    )

        return out


    def postprocess(self, accumulator):
        return accumulator


class TmplWriterUnit(ProcessingUnit):
    r"""The unit processing wrapper of the second step (calculate the coastline and derive the fit template"""

    def __init__(self, global_cfg, job_name='3_tmpl_writer', job_name_step1='1_mc_reweight', job_name_step2='2_coastline', fileset=None, **kwargs):
        super().__init__(
            job_name=job_name,
            fileset=fileset,
            processor_cls=TmplWriterCoffeaProcessor,
            processor_kwargs={'global_cfg': global_cfg},
            **kwargs,
        )
        self.global_cfg = global_cfg
        self.job_name_step1 = job_name_step1
        self.job_name_step2 = job_name_step2
        self.outputdir = os.path.join('output', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name)
        self.outputdir_step1 = os.path.join('output', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name_step1)
        self.outputdir_step2 = os.path.join('output', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name_step2)
        self.webdir = os.path.join('web', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name)
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
        if not os.path.exists(self.webdir):
            os.makedirs(self.webdir)
            

    def preprocess(self):
        def _check_exists(dirpath, filename, jobname):
            if not os.path.isfile(os.path.join(dirpath, filename)):
                _logger.exception('Cannot find ' + os.path.join(dirpath, filename) + '\n' \
                    + 'Please run step ' + jobname + ' first.')
                raise

        ## 1. Read the reweight map from step1, sfBDT weight map from step2
        _check_exists(self.outputdir_step1, 'hist.json', self.job_name_step1)
        _check_exists(self.outputdir_step2, 'sfbdt_hist.json', self.job_name_step2)
        with open(os.path.join(self.outputdir_step1, 'hist.json')) as f:
            self.weight_map = json.load(f)
        with open(os.path.join(self.outputdir_step2, 'sfbdt_hist.json')) as f:
            self.sfbdt_weight_map = json.load(f)
        
        ## 2. Read the xtagger map and sfBDT coastline from step2
        _check_exists(self.outputdir_step2, 'xtagger_map.pickle', self.job_name_step2)
        _check_exists(self.outputdir_step2, 'coastline_map.pickle', self.job_name_step2)
        with open(os.path.join(self.outputdir_step2, 'xtagger_map.pickle'), 'rb') as f:
            self.xtagger_map = pickle.load(f)
        with open(os.path.join(self.outputdir_step2, 'coastline_map.pickle'), 'rb') as f:
            self.coastline_map = pickle.load(f)

        ## 3. Put into arguments to initialize the coffea processor
        self.processor_kwargs.update(
            weight_map=self.weight_map,
            xtagger_map=self.xtagger_map,
            coastline_map=self.coastline_map,
            sfbdt_weight_map=self.sfbdt_weight_map,
        )


    def postprocess(self):

        # 1. Store raw output
        with open(os.path.join(self.outputdir, 'result.pickle'), 'wb') as fw:
            pickle.dump(self.result, fw)
        
        # 2. Write fit template
        if not self.global_cfg.skip_tmpl_writing:

            _logger.info("[Postprocess]: Writing the template for fit.")
            p = self.processor_instance
            writer_handler = StandaloneMultiThreadedUnit(workers=self.workers, use_unordered_mapping=True)
            args = SimpleNamespace(write_untypes=p.write_untypes, outputdir=self.outputdir)

            for wp in p.wps: # WP loop
                for ipt, (ptmin, ptmax) in enumerate(zip(p.pt_edges[:-1], p.pt_edges[1:])): # pt loop
                    # now all histogram infos are stored in the high-dim coffea hist
                    # we will use that to write all corresponding templates
                    bhs = {key: self.result[key].to_boost() for key in self.result if key.startswith(f'h_pt{ptmin}to{ptmax}_{wp}')} # used hists for one tasks
                    nbdt = len(self.coastline_map[ipt]['levels'])
                    writer_handler.book((args, bhs, wp, (ptmin, ptmax), nbdt))

            # run the tasks concurrently
            result = writer_handler.run(concurrent_tmpl_writing_unit)

            # also pickling the templates directly in the boost histogram format
            self.tmpl_hist = {}
            self.tmpl_fitpath = []
            for hs, fps in result:
                self.tmpl_hist.update(hs)
                self.tmpl_fitpath.extend(fps)
            with open(os.path.join(self.outputdir, 'tmpl_hist.pickle'), 'wb') as fw:
                pickle.dump(self.tmpl_hist, fw)
            with open(os.path.join(self.outputdir, 'tmpl_fitpath.pickle'), 'wb') as fw:
                pickle.dump(self.tmpl_fitpath, fw)


    def make_webpage(self):

        # Make inclusive plots (sfBDT, xtagger, etc.)
        if not self.global_cfg.skip_inclusive_plot_writing:
            _logger.info('[Make webpage]: Making inclusive plots.')

            # Init the web maker
            web = WebMaker(self.job_name)
            web.add_h1("Inclusive plots (pass + fail region)")
            web.add_text("Left to right: specified jet pT range:")
            pt_edges = self.global_cfg.pt_edges + [100000]
            web.add_text(', '.join([f'({ptmin}, {ptmax})' for ptmin, ptmax in zip(pt_edges[:-1], pt_edges[1:])]))
            web.add_text()
            web.add_text("Up to bottom: coastline cut level dependent on pT. The bottom plot is for inclusive i.e. no coastline selection")
            web.add_text()

            # make plots concurrently
            p = self.processor_instance
            plotter_handler = StandaloneMultiThreadedUnit(workers=self.workers, use_unordered_mapping=True)
            flvbin = ['flvL', 'flvB', 'flvC']
            iflvbin_order = [0, 2, 1] if self.global_cfg.type=='bb' else [0, 1, 2] if self.global_cfg.type=='cc' else [2, 1, 0] # e.g. flvL, flvC, flvB for bb
            color_mc = sns.color_palette('cubehelix', 3)
            year, lumi = self.global_cfg.year, self.global_cfg.lumi_dict[self.global_cfg.year]

            
            for var in p.incl_var_dict:
                xlabel, _ = p.incl_var_dict[var]
                for ipt, (ptmin, ptmax) in enumerate(zip(p.pt_edges[:-1], p.pt_edges[1:])): # pt loop
                    args = SimpleNamespace(
                        color_mc=color_mc, year=year, lumi=lumi,
                        flvbin=flvbin, iflvbin_order=iflvbin_order,
                        use_helvetica=self.global_cfg.use_helvetica,
                        logmsv_div_by_binw=self.global_cfg.logmsv_div_by_binw,
                    )
                    h = self.result[f'hinc_{var}_pt{ptmin}to{ptmax}'].to_boost()
                    nbdt = len(self.coastline_map[ipt]['levels'])
                    plot_args = {'ylog': True} if var == 'xtagger' else {}
                    plotter_handler.book((args, self.webdir, h, (var, xlabel), (ptmin, ptmax), nbdt, plot_args))
            
            plotter_handler.run(concurrent_inclusive_plot_writing_unit) # plots are stored in each concurrent task


            # collect plots on webpage
            for var in p.incl_var_dict:
                xlabel, _ = p.incl_var_dict[var]
                web.add_h2(xlabel)
                nbdt = len(self.coastline_map[0]['levels'])
                for ibdt in range(nbdt + 1): # coastline loop (column)
                    if ibdt != nbdt:
                        web.add_text(f'coastline #{ibdt}')
                    else:
                        web.add_text('inclusive, no coastline selection')
                    web.add_text()
                    for ipt, (ptmin, ptmax) in enumerate(zip(p.pt_edges[:-1], p.pt_edges[1:])): # pt loop (row)
                        web.add_figure(self.webdir, src=f'incl_{var}_csl{ibdt}_{year}_pt{ptmin}to{ptmax}.png', title=f'pT ({ptmin}, {ptmax}), coastline #{ibdt}')
                    web.add_text() # go to next row

            web.write_to_file(self.webdir)


def concurrent_tmpl_writing_unit(arg):
    r"""Unit concurrent task to launch the rest for-loop in the template writing.
    """
    args, bhs, wp, (ptmin, ptmax), nbdt = arg
    hs_stored, fitpath_stored = {}, []
    assert nbdt <= 10 and nbdt % 2 == 1
    bdtlist = [(i, i) for i in range(nbdt)] + [(i, j) for i in range(nbdt) for j in range(nbdt) if j != i]
    for ibdt, jbdt in bdtlist: # bdt loop. First process ibdt == jbdt, then copy the templates to other cases
        # fitpath is the nested folder for a unit fit point
        fitpath = os.path.join(args.outputdir, 'cards', wp, f'pt{ptmin}to{ptmax}', f'bdt{ibdt}{jbdt}')
        fitpath_stored.append(fitpath)
        if not os.path.exists(fitpath):
            os.makedirs(fitpath)
        for ipasswp, passwp in zip(range(2), ['fail', 'pass']): # store into input_fail.root or input_pass.root
            filepath = os.path.join(fitpath, f'inputs_{passwp}.root') # file to create

            with uproot3.recreate(filepath) as fw:
                for w_untype in args.write_untypes: # uncertainty type

                    # if not w_untype.startswith('sfBDTRwgt'):
                    #     tot_fac = 1.
                    # else:
                    #     # before slicing hists and storing to pass/fail root file, calculate the scale factors to reweight all MC to data
                    #     h_incl_mc = get_unit_template(bhs, wp, (ptmin, ptmax), ibdt, w_untype, None, None, is_mc=True, is_incl=True)
                    #     h_incl_data = get_unit_template(bhs, wp, (ptmin, ptmax), ibdt, w_untype, None, None, is_mc=False, is_incl=True)
                    #     # total factor will be used to fill every MC hist; bin-wise factors used in the fitRwgtVar unce type 
                    #     tot_fac = sum(h_incl_data.values(flow=True)) / sum(h_incl_mc.values(flow=True))
                    #     # print(tot_fac)

                    # We do no normalize the inclusive (pass+fail) templates before fit!
                    tot_fac = 1.
                    
                    if ibdt == jbdt:
                        for iflv, flv in zip(range(3), ['flvL', 'flvB', 'flvC']): # multiple hists in a root file
                            cat = flv if w_untype == 'nominal' else (flv + '_' + w_untype)

                            # now, access the target hist by: h[bh.loc(some cat), iflv, ipasswp, bdt_indices[ibdt], :] 
                            # get MC templates: pass in the additional options which can be used for some unce type if needed
                            is_mc = True
                            h_fit = get_unit_template(bhs, wp, (ptmin, ptmax), ibdt, w_untype, ipasswp, iflv, is_mc=is_mc, is_incl=False,
                                                additional_options={'factor': tot_fac, 'hists': hs_stored})
                            h_fit = fix_bh(h_fit)
                            hs_stored[(wp, (ptmin, ptmax), ibdt, w_untype, ipasswp, iflv, is_mc)] = h_fit
                            fw[cat] = bh_to_uproot3(h_fit)
                            # store data_obs for nominal hist
                        if w_untype == 'nominal':
                            is_mc = False
                            h_fit_data = get_unit_template(bhs, wp, (ptmin, ptmax), ibdt, w_untype, ipasswp, iflv, is_mc=is_mc, is_incl=False)
                            h_fit_data = fix_bh(h_fit_data)
                            hs_stored[(wp, (ptmin, ptmax), ibdt, w_untype, ipasswp, iflv, is_mc)] = h_fit_data
                            fw['data_obs'] = bh_to_uproot3(h_fit_data)
                    else: # ibdt != jbdt
                        # copy the existing root file
                        if passwp == 'fail':
                            shutil.copy(
                                os.path.join(args.outputdir, 'cards', wp, f'pt{ptmin}to{ptmax}', f'bdt{jbdt}{jbdt}', f'inputs_{passwp}.root'),
                                filepath
                            )
                        elif passwp == 'pass':
                            shutil.copy(
                                os.path.join(args.outputdir, 'cards', wp, f'pt{ptmin}to{ptmax}', f'bdt{ibdt}{ibdt}', f'inputs_{passwp}.root'),
                                filepath
                            )
    # when all finish, return the stored boost histograms for bookkeeping
    return (hs_stored, fitpath_stored)


def concurrent_inclusive_plot_writing_unit(arg):
    r"""Unit concurrent task to make inclusive plots based on the coffea result histogram.
    """
    args, webdir, h, (var, xlabel), (ptmin, ptmax), nbdt, plot_args = arg
    bdt_indices = [bh.underflow] + list(range(nbdt - 1)) + [bh.overflow]
    edges = h.axes[-1].edges
    mpl.use('Agg') # standalone job should individually specify the mpl mode...
    for ibdt in range(nbdt + 1): # different tightness of coastline selection
        h_data = bh.sum([ # sum over ibdt
            h[bh.loc('jetht'), :, bdt_indices[iibdt], :] for iibdt in range(ibdt + 1)
        ])
        h_mc = bh.sum([ # sum over ibdt
            bh.sum([ # sum over category
                h[bh.loc(h.axes[0].value(i)), :, bdt_indices[iibdt], :] \
                    for i in range(h.axes[0].size) if h.axes[0].value(i) != 'jetht'
            ]) for iibdt in range(ibdt + 1)
        ])
        values_mc_list = [h_mc[iflv, :].values() for iflv in args.iflvbin_order]
        values_data = h_data[bh.sum, :].values()
        yerr_mctot = np.sqrt(h_mc[bh.sum, :].variances())
        yerrlo_data = yerrhi_data = np.sqrt(h_data[bh.sum, :].variances())
        plot_text = '$p_T$: [{ptmin}, {ptmax}) GeV'.format(ptmin=ptmin, ptmax=ptmax if ptmax != 100000 else '+∞')
        plot_subtext = f'Coastline index: {ibdt+1} / {nbdt}' if ibdt != nbdt else 'Inclusive (no coastline selection)'
        store_name = os.path.join(webdir, f'incl_{var}_csl{ibdt}_{args.year}_pt{ptmin}to{ptmax}')

        if var == 'logmsv' and args.logmsv_div_by_binw:
            for ar in values_mc_list + [yerr_mctot, values_data, yerrlo_data, yerrhi_data]:
                ar[0] /= 4.; ar[1] /= 4.
                ar[-2] /= 7.; ar[-1] /= 7.
            make_generic_mc_data_plots(
                edges, values_mc_list, yerr_mctot, values_data, yerrlo_data, yerrhi_data,
                [f'MC ({args.flvbin[iflv]})' for iflv in args.iflvbin_order], args.color_mc,
                xlabel, 'Events / 0.1', args.year, args.lumi,
                use_helvetica=args.use_helvetica, plot_args=plot_args,
                plot_text=plot_text, plot_subtext=plot_subtext,
                store_name=store_name
            )
        else:
            make_generic_mc_data_plots(
                edges, values_mc_list, yerr_mctot, values_data, yerrlo_data, yerrhi_data,
                [f'MC ({args.flvbin[iflv]})' for iflv in args.iflvbin_order], args.color_mc,
                xlabel, 'Events / bin', args.year, args.lumi,
                use_helvetica=args.use_helvetica, plot_args=plot_args,
                plot_text=plot_text, plot_subtext=plot_subtext,
                store_name=store_name
            )


def get_unit_template(bhs, wp, pt_lim, ibdt, w_untype, ipasswp, iflv, is_mc=True, is_incl=False, additional_options={}):
    r"""Get the specific 1D fit template from the high-dim boost histogram results
    Additional arguments:
        is_mc: if derive MC or data template. If data, untype, and iflv will be disabled
        is_incl: if derive inclusive histogram on ipasswp and iflv.
    """

    def default(w_untype_=w_untype, ipasswp_=ipasswp, iflv_=iflv, is_mc_=is_mc, is_incl_=is_incl):
        # 5 regions for sfBDT coastline selection, from tight to loose. Note that the template for fit is accumulative sum on this axis
        bdt_indices = [bh.underflow] + list(range(ibdt))
        h = bhs[f'h_pt{pt_lim[0]}to{pt_lim[1]}_{wp}_{w_untype_}']
        if is_mc_:
            return bh.sum([ # sum over ibdt
                bh.sum([ # sum over category
                    h[bh.loc(h.axes[0].value(i)), iflv_ if not is_incl_ else bh.sum, ipasswp_ if not is_incl_ else bh.sum, bdt_indices[iibdt], :] \
                        for i in range(h.axes[0].size) if h.axes[0].value(i) != 'jetht'
                ]) for iibdt in range(ibdt + 1)
            ])
        else:
            return bh.sum([ # sum over ibdt
                h[bh.loc('jetht'), 0 if not is_incl_ else bh.sum, ipasswp_ if not is_incl_ else bh.sum, bdt_indices[iibdt], :] for iibdt in range(ibdt + 1)
            ])

    assert w_untype in ['nominal', 'puUp', 'puDown', 'l1PreFiringUp', 'l1PreFiringDown', 'jesUp', 'jesDown', 'jerUp', 'jerDown', \
        'fracBBUp', 'fracBBDown', 'fracCCUp', 'fracCCDown', 'fracLightUp', 'fracLightDown', \
        'psWeightIsrUp', 'psWeightIsrDown', 'psWeightFsrUp', 'psWeightFsrDown', \
        'sfBDTRwgtUp', 'sfBDTRwgtDown', 'fitVarRwgtUp', 'fitVarRwgtDown'], "Unrecognized unce type for writing templates."

    # not special handling for retreiving data template
    if not is_mc:
        return default(w_untype_='nominal')

    # for these unce type: simply retrive the histograms from coffea output
    if w_untype in ['nominal', 'puUp', 'puDown', 'l1PreFiringUp', 'l1PreFiringDown', 'jesUp', 'jesDown', 'jerUp', 'jerDown', 'psWeightIsrUp', 'psWeightIsrDown', 'psWeightFsrUp', 'psWeightFsrDown', 'sfBDTRwgtUp']:
        h_out = default()
        if is_incl:
            return h_out
        else:
            # multiply the overall factor passed in from inclusive plots
            assert 'factor' in additional_options
            h_out = scale_bh(h_out, additional_options['factor'])
            return h_out

    # below are special treatments. Note that we must have is_mc==True if code goes here
    elif w_untype in ['fracBBUp', 'fracBBDown', 'fracCCUp', 'fracCCDown', 'fracLightUp', 'fracLightDown']:
        target_iflv = 0 if 'fracLight' in w_untype else 1 if 'fracBB' in w_untype else 2
        ud = 'Up' if w_untype.endswith('Up') else 'Down'
        if is_incl:
            # fetch the corresponding flavour template from the fracBCLUp/Down template
            # e.g. fracBBUp has flvB stored in the fracBCLUp histogram from coffea, and others (flvC, flvL) just in the nominal histogram from coffea
            h_flv_passwp_list = [default(w_untype_='nominal' if iiflv != target_iflv else ('fracBCL' + ud), ipasswp_=iipasswd, iflv_=iiflv, is_incl_=False) for iipasswd in [0, 1] for iiflv in [0, 1, 2]]
            return bh.sum(h_flv_passwp_list)
        else:
            h_out = default(w_untype_='nominal' if iflv != target_iflv else ('fracBCL' + ud))
            # multiply the overall factor
            assert 'factor' in additional_options
            h_out = scale_bh(h_out, additional_options['factor'])
            return h_out


    elif w_untype == 'fitVarRwgtUp':
        if is_incl:
            return default(w_untype_='nominal') # dummy here
        else:
            # reweight the inclusive MC (all flavour) to data to extract the binwise SF, then apply on each flavour template
            h_incl_mc, h_incl_data = default(w_untype_='nominal', is_mc_=True, is_incl_=True), default(w_untype_='nominal', is_mc_=False, is_incl_=True)
            binwise_factors = h_incl_data.values(flow=True) / np.maximum(h_incl_mc.values(flow=True), 1e-20)
            h_nom = default(w_untype_='nominal')
            return scale_bh(h_nom, binwise_factors)
    
    elif w_untype in ['sfBDTRwgtDown', 'fitVarRwgtDown']:
        # no need to derive the inclusive templates for the overall factor
        # princeple is down = 2 * nominal - up, where nominal and up has their corresponding factors
        if is_incl:
            return default(w_untype_='nominal') # dummy here
        else:
            # read nominal and up templated from stored template
            hs = additional_options.get('hists', None)
            w_untype_up = w_untype.replace('Down', 'Up')
            nom_args = (wp, pt_lim, ibdt, 'nominal', ipasswp, iflv, is_mc)
            up_args = (wp, pt_lim, ibdt, w_untype_up, ipasswp, iflv, is_mc) 
            assert nom_args in hs, f'Please process the nominal template before {w_untype}'
            assert up_args in hs, f'Please process the {w_untype_up} template before {w_untype}'
            h_up, h_nom = hs[up_args], hs[nom_args]
            h_down = h_up.copy()
            # calculate down template. Note we only care about bin values for the up/down template, not variances anymore
            for i in range(len(h_up.view(flow=True))):
                h_down.values(flow=True)[i] = 2 * h_nom.values(flow=True)[i] - h_up.values(flow=True)[i]
            return h_down