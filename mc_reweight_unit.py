"""
For step 1: reweight total MC to data due to the use of prescaled HT triggers.

"""

from coffea import processor, hist
import awkward as ak
import numpy as np
import uproot
import uproot3

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler 
mpl.use('Agg')
mpl.rcParams['axes.prop_cycle'] = cycler(color=['blue', 'red', 'green', 'violet', 'darkorange', 'black', 'cyan', 'yellow'])
import mplhep as hep
plt.style.use(hep.style.CMS)

from functools import partial
import pickle
import json
import os

from unit import ProcessingUnit
from utils.web_maker import WebMaker
from logger import _logger


class MCReweightCoffeaProcessor(processor.ProcessorABC):
    r"""The coffea processor for the reweighting step"""

    def __init__(self, global_cfg=None):
        self.global_cfg = global_cfg

        dataset = hist.Cat("dataset", "dataset")

        def linspace_bin(start, end, width=50):
            return hist.Bin('ht', 'ht', (end - start) // width, start, end)

        _hists = {}
        for suffix in ['', '_jesUp', '_jesDown', '_jerUp', '_jerDown']:
            _hists.update({
                f'ht_fj1_pt200to250{suffix}': hist.Hist('Counts', dataset, linspace_bin(250, 1250)),
                f'ht_fj1_pt250to300{suffix}': hist.Hist('Counts', dataset, linspace_bin(350, 1400)),
                f'ht_fj1_pt300to350{suffix}': hist.Hist('Counts', dataset, linspace_bin(400, 1600)),
                f'ht_fj1_pt350to400{suffix}': hist.Hist('Counts', dataset, linspace_bin(450, 1700)),
                f'ht_fj1_pt400to450{suffix}': hist.Hist('Counts', dataset, linspace_bin(500, 1800)),
                f'ht_fj1_pt450to500{suffix}': hist.Hist('Counts', dataset, linspace_bin(550, 1900)),
                f'ht_fj1_pt500to550{suffix}': hist.Hist('Counts', dataset, linspace_bin(600, 1900)),
                f'ht_fj1_pt550to600{suffix}': hist.Hist('Counts', dataset, linspace_bin(650, 2000)),
                f'ht_fj1_pt600to700{suffix}': hist.Hist('Counts', dataset, linspace_bin(700, 2100)),
                f'ht_fj1_pt700to800{suffix}': hist.Hist('Counts', dataset, linspace_bin(800, 2200)),
                f'ht_fj1_pt800to100000{suffix}': hist.Hist('Counts', dataset, linspace_bin(1000, 2400)),

                f'ht_fj2_pt200to250{suffix}': hist.Hist('Counts', dataset, linspace_bin(250, 1500)),
                f'ht_fj2_pt250to300{suffix}': hist.Hist('Counts', dataset, linspace_bin(350, 1600)),
                f'ht_fj2_pt300to350{suffix}': hist.Hist('Counts', dataset, linspace_bin(400, 1800)),
                f'ht_fj2_pt350to400{suffix}': hist.Hist('Counts', dataset, linspace_bin(450, 2000)),
                f'ht_fj2_pt400to450{suffix}': hist.Hist('Counts', dataset, linspace_bin(500, 2200)),
                f'ht_fj2_pt450to500{suffix}': hist.Hist('Counts', dataset, linspace_bin(550, 2400)),
                f'ht_fj2_pt500to550{suffix}': hist.Hist('Counts', dataset, linspace_bin(650, 2400)),
                f'ht_fj2_pt550to600{suffix}': hist.Hist('Counts', dataset, linspace_bin(750, 2400)),
                f'ht_fj2_pt600to700{suffix}': hist.Hist('Counts', dataset, linspace_bin(850, 2400)),
                f'ht_fj2_pt700to800{suffix}': hist.Hist('Counts', dataset, linspace_bin(1000, 2400)),
                f'ht_fj2_pt800to100000{suffix}': hist.Hist('Counts', dataset, linspace_bin(1200, 2400)),
            })
        _hists['cutflow'] = processor.defaultdict_accumulator(
            partial(processor.defaultdict_accumulator, int)
        )
        self._accumulator = processor.dict_accumulator(_hists)

    @property
    def accumulator(self):
        return self._accumulator


    def process(self, events):
        out = self.accumulator.identity()
        dataset = events.metadata['dataset']
        is_mc = dataset != 'jetht'

        presel = ak.numexpr.evaluate('passmetfilters & (' + '|'.join(self.global_cfg.hlt_branches[self.global_cfg.year]) + ')', events)
        custom_sel_fj1 = ak.numexpr.evaluate(self.global_cfg.custom_selection.replace('fj_x', 'fj_1'), events) if self.global_cfg.custom_selection is not None else ak.ones_like(presel, dtype=bool)
        custom_sel_fj2 = ak.numexpr.evaluate(self.global_cfg.custom_selection.replace('fj_x', 'fj_2'), events) if self.global_cfg.custom_selection is not None else ak.ones_like(presel, dtype=bool)
        events_fj1 = events[(presel) & (events.fj_1_is_qualified) & (custom_sel_fj1)]
        events_fj2 = events[(presel) & (events.fj_2_is_qualified) & (custom_sel_fj2)]
        lumi = self.global_cfg.lumi_dict[self.global_cfg.year]

        for ptmin, ptmax in self.global_cfg.rwgt_pt_bins:
            events_fj1_pt = events_fj1[(events_fj1.fj_1_pt >= ptmin) & (events_fj1.fj_1_pt < ptmax)]
            events_fj2_pt = events_fj2[(events_fj2.fj_2_pt >= ptmin) & (events_fj2.fj_2_pt < ptmax)]
            ht_nom_fj1 = events_fj1_pt.ht
            ht_nom_fj2 = events_fj2_pt.ht

            # Fill the qualified fj_1 and fj_2 events separately
            weight_nom_fj1 = ak.numexpr.evaluate(f'genWeight*xsecWeight*puWeight*l1PreFiringWeight*{lumi}', events_fj1_pt) if is_mc else ak.ones_like(events_fj1_pt.ht)
            weight_nom_fj2 = ak.numexpr.evaluate(f'genWeight*xsecWeight*puWeight*l1PreFiringWeight*{lumi}', events_fj2_pt) if is_mc else ak.ones_like(events_fj2_pt.ht)
            # Fill histograms for nominal case ('') and JES/JER cases; no
            for suffix in (['', '_jesUp', '_jesDown', '_jerUp', '_jerDown'] if is_mc else ['']):
                if suffix == '':
                    ht_fj1, ht_fj2 = ht_nom_fj1, ht_nom_fj2
                    weight_fj1 = weight_nom_fj1
                    weight_fj2 = weight_nom_fj2

                # the following PSWeight variation schemes are not considered now
                elif suffix == '_psWeightIsrUp':
                    ht_fj1, ht_fj2 = ht_nom_fj1, ht_nom_fj2
                    weight_fj1 = weight_nom_fj1 * events_fj1_pt.PSWeight[:,2]
                    weight_fj2 = weight_nom_fj2 * events_fj2_pt.PSWeight[:,2]
                elif suffix == '_psWeightIsrDown':
                    ht_fj1, ht_fj2 = ht_nom_fj1, ht_nom_fj2
                    weight_fj1 = weight_nom_fj1 * events_fj1_pt.PSWeight[:,0]
                    weight_fj2 = weight_nom_fj2 * events_fj2_pt.PSWeight[:,0]
                elif suffix == '_psWeightFsrUp':
                    ht_fj1, ht_fj2 = ht_nom_fj1, ht_nom_fj2
                    weight_fj1 = weight_nom_fj1 * events_fj1_pt.PSWeight[:,3]
                    weight_fj2 = weight_nom_fj2 * events_fj2_pt.PSWeight[:,3]
                elif suffix == '_psWeightFsrDown':
                    ht_fj1, ht_fj2 = ht_nom_fj1, ht_nom_fj2
                    weight_fj1 = weight_nom_fj1 * events_fj1_pt.PSWeight[:,1]
                    weight_fj2 = weight_nom_fj2 * events_fj2_pt.PSWeight[:,1]

                elif suffix in ['_jesUp', '_jesDown', '_jerUp', '_jerDown']:
                    suffix_to_branch = {'_jesUp': '_jesUncFactorUp', '_jesDown': '_jesUncFactorDn', '_jerUp': '_jerSmearFactorUp', '_jerDown': '_jerSmearFactorDn'}
                    # should use corrected ht and jet pt to fill histograms
                    pt_fj1_corr = events_fj1.fj_1_pt * events_fj1[f'fj_1{suffix_to_branch[suffix]}']
                    pt_fj2_corr = events_fj2.fj_2_pt * events_fj2[f'fj_2{suffix_to_branch[suffix]}']
                    events_fj1_pt_corr = events_fj1[(pt_fj1_corr >= ptmin) & (pt_fj1_corr < ptmax)]
                    events_fj2_pt_corr = events_fj2[(pt_fj2_corr >= ptmin) & (pt_fj2_corr < ptmax)]

                    ht_fj1, ht_fj2 = events_fj1_pt_corr[f'ht{suffix_to_branch[suffix]}'], events_fj2_pt_corr[f'ht{suffix_to_branch[suffix]}']
                    weight_fj1 = ak.numexpr.evaluate(f'genWeight*xsecWeight*puWeight*l1PreFiringWeight*{lumi}', events_fj1_pt_corr)
                    weight_fj2 = ak.numexpr.evaluate(f'genWeight*xsecWeight*puWeight*l1PreFiringWeight*{lumi}', events_fj2_pt_corr)
                out[f'ht_fj1_pt{ptmin}to{ptmax}{suffix}'].fill(
                    dataset=dataset,
                    ht=ht_fj1,
                    weight=weight_fj1,
                )
                out[f'ht_fj2_pt{ptmin}to{ptmax}{suffix}'].fill(
                    dataset=dataset,
                    ht=ht_fj2,
                    weight=weight_fj2,
                )

            out['cutflow'][dataset][f'fj1_pt{ptmin}to{ptmax}'] += len(events_fj1_pt)
            out['cutflow'][dataset][f'fj2_pt{ptmin}to{ptmax}'] += len(events_fj2_pt)

        return out


    def postprocess(self, accumulator):
        return accumulator


class MCReweightUnit(ProcessingUnit):
    r"""The unit processing wrapper of the MC reweighting step"""

    def __init__(self, global_cfg, job_name='1_mc_reweight', fileset=None, **kwargs):
        super().__init__(
            job_name=job_name,
            fileset=fileset,
            processor_cls=MCReweightCoffeaProcessor,
            processor_kwargs={'global_cfg': global_cfg},
            **kwargs,
        )
        self.global_cfg = global_cfg
        self.outputdir = os.path.join('output', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name)
        self.webdir = os.path.join('web', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name)
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
        if not os.path.exists(self.webdir):
            os.makedirs(self.webdir)


    def postprocess(self):

        _logger.info("[Postprocess]: Storing the reweighting histograms.")

        # Store raw output
        with open(os.path.join(self.outputdir, 'result.pickle'), 'wb') as fw:
            pickle.dump(self.result, fw)

        # Store cutflow numbers to json file
        with open(os.path.join(self.outputdir, 'cutflow.json'), 'w') as fw:
            json.dump(self.result['cutflow'], fw, indent=4)

        # Caculate and store reweighting values to json file
        hist_values = {}
        for suffix in ['', '_jesUp', '_jesDown', '_jerUp', '_jerDown']:
            for ptmin, ptmax in self.global_cfg.rwgt_pt_bins:
                h_mc_fj, h_data_fj = {}, {}
                for jetidx in ['fj1', 'fj2']:
                    ptbname = f'ht_{jetidx}_pt{ptmin}to{ptmax}'
                    h_data_fj[jetidx] = self.result[ptbname]['jetht'].project('ht')
                    h_mc_fj[jetidx] = sum([self.result[ptbname + suffix][sam].project('ht') for sam in self.fileset if sam != 'jetht'], h_data_fj[jetidx].identity())
                    # store hist into numerical values
                    _stored = {
                        'edges': h_data_fj[jetidx].axis('ht').edges().tolist(),
                        'h_data': h_data_fj[jetidx].to_boost().values(flow=True).tolist(),
                        'h_mc': h_mc_fj[jetidx].to_boost().values(flow=True).tolist(),
                        'h_w': np.clip(h_data_fj[jetidx].to_boost().values(flow=True) / np.maximum(h_mc_fj[jetidx].to_boost().values(flow=True), 1e-20), 0., 2.).tolist(),
                    }
                    hist_values[f'{jetidx}_pt{ptmin}to{ptmax}{suffix}'] = _stored

        with open(os.path.join(self.outputdir, 'hist.json'), 'w') as fw:
            json.dump(hist_values, fw, indent=4)


    def make_webpage(self):

        _logger.info('[Make webpage]: Making the reweight histograms then put on the webpage.')

        if not hasattr(self, 'result') or self.result is None:
            self.load_pickle('result')

        # Configure mplhep
        if self.global_cfg.use_helvetica == True or (self.global_cfg.use_helvetica == 'auto' and any(['Helvetica' in font for font in mpl.font_manager.findSystemFonts()])):
            plt.style.use({"font.sans-serif": 'Helvetica'})
        year, lumi = self.global_cfg.year, self.global_cfg.lumi_dict[self.global_cfg.year]
 
        # Init the web maker
        web = WebMaker(self.job_name)
        web.add_h1("Reweight factors in 3D bins")
        web.add_text("Left to right: factors shown in 2D (HT, jetidx) for each jet pT range:")
        web.add_text(', '.join([f'({ptmin}, {ptmax})' for ptmin, ptmax in self.global_cfg.rwgt_pt_bins]))
        web.add_text()

        # Make plots and originze in the webpage
        with open(os.path.join(self.outputdir, 'hist.json')) as f:
            hist_values = json.load(f)
        for ptmin, ptmax in self.global_cfg.rwgt_pt_bins:
            # each pt bin will result in a plot
            h_fj1 = hist_values[f'fj1_pt{ptmin}to{ptmax}']
            h_fj2 = hist_values[f'fj2_pt{ptmin}to{ptmax}']
            # make plot
            f = plt.figure(figsize=(10, 10))
            gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.04) 
            ax, ax1 = f.add_subplot(gs[0]), f.add_subplot(gs[1])
            hep.cms.label(data=True, llabel='Preliminary', year=year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % lumi, fontname='sans-serif')
            for i, cm, cd, hist in zip(['1', '2'], ['blue', 'red'], ['royalblue', 'lightcoral'], [h_fj1, h_fj2]):
                # print(hist['h_mc'], hist['h_data'])
                bins = [0] + hist['edges'] + [2500]
            
                hep.histplot(hist['h_mc'], bins=bins, label=f'Jet {i} (MC)', color=cm, ax=ax)
                hep.histplot(hist['h_data'], bins=bins, label=f'Jet {i} (Data)', color=cd, linestyle='--', ax=ax)
                hep.histplot(hist['h_w'], bins=bins, label=f'Jet {i}', color=cm, ax=ax1)

            ax.set_xlim(0, 2500); ax.set_xticklabels([]); 
            ax.set_yscale('log'); ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*10.); ax.set_ylabel('Events', ha='right', y=1.0)
            ax.text(0.03, 0.91, '$p_T$: [{ptmin}, {ptmax}) GeV'.format(ptmin=ptmin, ptmax=ptmax if ptmax != 100000 else '+∞'), transform=ax.transAxes, fontweight='bold', fontsize=21)
            ax.legend()
            ax1.set_xlim(0, 2500); ax1.set_xlabel('$H_{T}$ [GeV]', ha='right', x=1.0);
            ax1.set_yscale('log'); ax1.set_ylim(5e-3, 2e0); ax1.set_ylabel('Rwgt factor', ha='right', y=1.0); ax1.set_yticks([1e-2, 1e-1, 1e0, 1e1]);
            ax1.legend()
            ax1.plot([0, 2500], [1, 1], 'k:')

            plt.savefig(os.path.join(self.webdir, f'rwgtfac_{year}_pt{ptmin}to{ptmax}.png'))
            plt.savefig(os.path.join(self.webdir, f'rwgtfac_{year}_pt{ptmin}to{ptmax}.pdf'))
            plt.close()

            web.add_figure(self.webdir, src=f'rwgtfac_{year}_pt{ptmin}to{ptmax}.png', title=f'pT ({ptmin}, {ptmax})')
        
        web.write_to_file(self.webdir)