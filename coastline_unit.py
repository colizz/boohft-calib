"""
For step 2: calculate the sfBDT coastline on the target transformed tagger.

"""

from coffea import processor, hist
import awkward as ak
import numpy as np
import uproot
import uproot3
import scipy

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler 
mpl.use('Agg')
mpl.rcParams['axes.prop_cycle'] = cycler(color=['blue', 'red', 'green', 'violet', 'darkorange', 'black', 'cyan', 'yellow'])

import boost_histogram as bh
import mplhep as hep
plt.style.use(hep.style.CMS)

from functools import partial
import pickle
import shutil
import json
import os

from unit import ProcessingUnit
from utils.web_maker import WebMaker
from utils.tools import lookup_pt_based_weight, parse_tagger_expr, eval_expr
from utils.fast_splines import interp2d
from logger import _logger


class CoastlineCoffeaProcessor(processor.ProcessorABC):
    r"""The coffea processor for the coastline and template writing step"""

    def __init__(self, global_cfg=None, weight_map=None, xtagger_map=None, **kwargs):
        self.global_cfg = global_cfg
        self.weight_map = weight_map
        self.xtagger_map = xtagger_map
        self.nbin2d = kwargs.pop('nbin2d', 200)
        self.pt_reweight_edges = [edge[0] for edge in global_cfg.rwgt_pt_bins]

        self.tagger_expr = parse_tagger_expr(global_cfg.tagger_name_replace_map, global_cfg.tagger.expr)
        self.lookup_mc_weight = partial(lookup_pt_based_weight, self.weight_map, self.pt_reweight_edges, jet_var_maxlimit=2500)

        dataset = hist.Cat("dataset", "dataset")

        # fine 2D grids on tagger-sfBDT to derive the coastline
        pt_bin = hist.Bin('pt', 'pt', list(global_cfg.pt_edges))
        sfbdt_grid = hist.Bin('sfbdt', 'sfbdt', self.nbin2d, 0., 1.)
        xtagger_grid = hist.Bin('xtagger', 'xtagger', self.nbin2d, 0., 1.) # transformed tagger bin
        h2d_grid = hist.Hist('Counts', dataset, pt_bin, sfbdt_grid, xtagger_grid)

        # sfBDT 1D hist to derive data/MC discrepancy used for uncertainty
        jetidx_cat = hist.Cat('jetidx', 'jetidx')
        pt_finebin = hist.Bin('pt', 'pt', list(self.pt_reweight_edges))
        sfbdt_bin = hist.Bin('sfbdt', 'sfbdt', 50, 0., 1.)
        h_sfbdt = hist.Hist('Counts', dataset, jetidx_cat, pt_finebin, sfbdt_bin)

        self._accumulator = processor.dict_accumulator({
            'h2d_grid': h2d_grid,
            'h_sfbdt': h_sfbdt,
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
            events_fj = events[(presel) & (events[f'fj_{i}_is_qualified'])]

            # calculate weights and flavour variables
            if is_mc:
                # calculate the MC-to-data weigts only for MC
                mc_weight = self.lookup_mc_weight(f'fj{i}', events_fj[f'fj_{i}_pt'], events_fj['ht'])
                weight = ak.numexpr.evaluate(f'genWeight*xsecWeight*puWeight*{lumi}', events_fj) * mc_weight
                assert self.global_cfg.type in ['bb', 'cc', 'qq'], "Calibration type must be 'bb', 'cc', or 'qq'."
                if self.global_cfg.type == 'bb':
                    flv_sel = ak.numexpr.evaluate(f'fj_{i}_nbhadrons >= 1', events_fj)
                elif self.global_cfg.type == 'cc':
                    flv_sel = ak.numexpr.evaluate(f'(fj_{i}_nbhadrons == 0) & (fj_{i}_nchadrons >= 1)', events_fj)
                elif self.global_cfg.type == 'qq':
                    flv_sel = ak.numexpr.evaluate(f'(fj_{i}_nbhadrons == 0) & (fj_{i}_nchadrons == 0)', events_fj)
            else:
                weight = ak.ones_like(events_fj.ht)

            # fill into histograms for each WP (range choices on tagger), MC only, flavour selection applied
            if is_mc:
                tagger_flv_sel = ak.numexpr.evaluate(self.tagger_expr.replace('fj_x', f'fj_{i}'), events_fj[flv_sel])
                xtagger_flv_sel = self.xtagger_map(tagger_flv_sel)
                out[f'h2d_grid'].fill(
                    dataset=dataset,
                    pt=events_fj[f'fj_{i}_pt'][flv_sel],
                    sfbdt=events_fj[f'fj_{i}_sfBDT'][flv_sel],
                    xtagger=xtagger_flv_sel,
                    weight=weight[flv_sel],
                )

            # fill the sfbdt histograms for all MC and data, without flavour selection
            out[f'h_sfbdt'].fill(
                dataset=dataset,
                jetidx=i,
                pt=events_fj[f'fj_{i}_pt'],
                sfbdt=events_fj[f'fj_{i}_sfBDT'],
                weight=weight,
            )

        return out


    def postprocess(self, accumulator):
        return accumulator


class CoastlineUnit(ProcessingUnit):
    r"""The unit processing wrapper of the second step (calculate the coastline and derive the fit template"""

    def __init__(self, global_cfg, job_name='2_coastline', job_name_step1='1_mc_reweight', fileset=None, **kwargs):
        # coastline variables
        self.nbin2d = kwargs.pop('nbin2d', 200)

        super().__init__(
            job_name=job_name,
            fileset=fileset,
            processor_cls=CoastlineCoffeaProcessor,
            processor_kwargs={ # processor_kwargs will be updated after preprocessing
                'global_cfg': global_cfg,
                'nbin2d': self.nbin2d,
            },
            **kwargs,
        )
        self.global_cfg = global_cfg
        self.job_name_step1 = job_name_step1
        self.outputdir = os.path.join('output', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name)
        self.outputdir_step1 = os.path.join('output', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name_step1)
        self.webdir = os.path.join('web', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name)
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
        if not os.path.exists(self.webdir):
            os.makedirs(self.webdir)


    def preprocess(self):
        ## 1. Obtain tagger shape from the provided signal analysis ROOT file
        anatree = self.global_cfg.main_analysis_tree

        if not hasattr(anatree, 'provide_tagger_array') or anatree.provide_tagger_array:
            # read the tagger array from the main analysis signal tree
            self.provide_tagger_array = True
            df = uproot.lazy(anatree.path.replace("$YEAR", str(self.global_cfg.year)) + ':' + anatree.treename)
            anatree_selection = eval_expr(anatree.selection, df)

            tagger = eval_expr(anatree.tagger, df)[anatree_selection]
            weight = eval_expr(anatree.weight, df)[anatree_selection]

            ## Transfrom the tagger to uniform distribution
            tmin, tmax = self.global_cfg.tagger.span

            nbin_hist = 1000
            hist = bh.Histogram(bh.axis.Regular(nbin_hist, tmin, tmax), storage=bh.storage.Weight())
            hist.fill(tagger, weight=weight)

        else:
            # use provided histogram as the direct template
            self.provide_tagger_array = False
            tmin, tmax = self.global_cfg.tagger.span
            hist = uproot.open(anatree.path.replace("$YEAR", str(self.global_cfg.year)) + ':' + anatree.histname).to_boost()
            nbin_hist = hist.axes[0].size
            assert tmin == hist.axes[0].edges[0] and tmax == hist.axes[0].edges[-1], \
                f"Provide tagger shape histogram has range [{hist.axes[0].edges[0]}, {hist.axes[0].edges[-1]}] which does not match with the tagger span."

        # cumulative distribution of the tagger:
        x = np.linspace(tmin, tmax, nbin_hist + 1)
        y = np.maximum(hist.values(), 1e-5)
        y_cum = np.insert(np.cumsum(y / sum(y)), 0, 0.)
        # plt.plot(x, y_cum); plt.show()

        # the tagger tranformation map is derived from the cumulative sum
        self.xtagger_map = scipy.interpolate.interp1d(x, y_cum, kind='cubic')

        if self.provide_tagger_array:
            # also store the signal hist info for later plot making
            xtagger = self.xtagger_map(tagger)
            hist = bh.Histogram(bh.axis.Regular(50, tmin, tmax), storage=bh.storage.Weight())
            hist.fill(tagger, weight=weight)
            xhist = bh.Histogram(bh.axis.Regular(50, 0., 1.), storage=bh.storage.Weight())
            xhist.fill(xtagger, weight=weight)
            self.signal_hist_info = {
                't_x': x, 't_y': y_cum, 'hist': hist, 'xhist': xhist
            }
        else:
            self.signal_hist_info = {
                't_x': x, 't_y': y_cum, 'hist': hist, 'xhist': None
            }

        with open(os.path.join(self.outputdir, 'xtagger_map.pickle'), 'wb') as fw:
            pickle.dump(self.xtagger_map, fw)

        ## 2. Read the reweight map
        if self.global_cfg.reuse_mc_weight_from_routine is not None:
            if os.path.exists(self.outputdir_step1):
                if not os.path.isdir(self.outputdir_step1):
                    _logger.exception(f"The directory {self.outputdir_step1} exists. This should not happen as you have specified 'reuse_mc_weight_from_routine'.")
                    raise
            else:
                # create the soft link for output/1_mc_reweight and copy web/1_mc_reweight
                os.symlink(
                    os.path.join('..', self.global_cfg.reuse_mc_weight_from_routine + '_' + str(self.global_cfg.year), self.job_name_step1),
                    self.outputdir_step1
                )
                shutil.copytree(
                    os.path.join('web', self.global_cfg.reuse_mc_weight_from_routine + '_' + str(self.global_cfg.year), self.job_name),
                    os.path.join('web', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name)
                )
        if not os.path.isfile(os.path.join(self.outputdir_step1, 'hist.json')):
            _logger.exception('Cannot find ' + os.path.join(self.outputdir_step1, 'hist.json') + '\n' \
                + 'Please run the previous step ' + self.job_name_step1 + ' first.')
            raise

        with open(os.path.join(self.outputdir_step1, 'hist.json')) as f:
            self.weight_map = json.load(f)

        ## 3. Put into arguments to initialize the coffea processor
        self.processor_kwargs.update(
            weight_map=self.weight_map,
            xtagger_map=self.xtagger_map,
        )


    def postprocess(self):

        _logger.info("[Postprocess]: Storing the sfBDT coastline histograms.")

        # 1. Store raw output
        with open(os.path.join(self.outputdir, 'result.pickle'), 'wb') as fw:
            pickle.dump(self.result, fw)
        
        # 2. Derive and store the sfBDT coastline
        self.coastline_map = []
    
        # the xtagger-sfBDT map for all pT ranges (under/overflow included)
        bh2d_pts = self.result['h2d_grid'].integrate('dataset').to_boost()
        for ipt in range(1, len(self.global_cfg.pt_edges) + 1):
            arr2d = bh2d_pts.view(flow=True).value[ipt][1:-1, 1:-1].T # (dim_tagger, dim_sfbdt)
            arr2d_norm = arr2d / np.sum(arr2d)

            # accumulate 2D hist on the y-axis (sfBDT)
            arr2d_cum = np.cumsum(arr2d_norm[:, ::-1], axis=1)[:, ::-1]

            # extend the y-limit on sfBDT a bit to rescue from the gaussian filter
            step = 1./ self.nbin2d
            y_ex = 0.04
            nstep_extend = int(y_ex / step)
            x = y = np.arange(step/2, 1. + step/2, step) # 200 bins correspond to the hist

            # smear the 2d hist with gaussian filter
            arr2d_cum_expend = np.zeros((arr2d_cum.shape[0], arr2d_cum.shape[1] + nstep_extend))
            arr2d_cum_expend[:, :-nstep_extend] = arr2d_cum
            arr2d_cum_smeared = scipy.ndimage.gaussian_filter(arr2d_cum_expend, sigma=5)[:, :-nstep_extend]
            arr2d_cum_smeared[:, -1] = 0.

            # define end point for the contour
            level_en = arr2d_cum_smeared[int(0.6 * len(x)), 0]  # at point (0.6, 0)
            levels = np.linspace(0, level_en, 12)[3:]

            _logger.debug(f'Calculated coastline contour levels for pT bin {ipt}: {str(levels)}')

            # fast-spline the 2d smeared hist
            fspline = interp2d(x, y, arr2d_cum_smeared)

            self.coastline_map.append({'arr2d': arr2d, 'arr2d_cum_smeared': arr2d_cum_smeared, 'fspline': fspline, 'levels': levels})
        
        with open(os.path.join(self.outputdir, 'coastline_map.pickle'), 'wb') as fw:
            pickle.dump(self.coastline_map, fw)

        # 3. Store the sfBDT reweight hist to json file
        hist_values = {}
        h_sfbdt = self.result['h_sfbdt'].to_boost()
        for ipt, (ptmin, ptmax) in enumerate(self.global_cfg.rwgt_pt_bins):
            for i in '12':
                sam_axes = h_sfbdt.axes[0]
                pt_loc = ipt if ipt != len(self.global_cfg.rwgt_pt_bins)-1 else bh.overflow
                # data and MC sfBDT 1D histogram
                h_data_fj = h_sfbdt[bh.loc('jetht'), bh.loc(i), pt_loc, :]
                h_mc_fj = bh.sum(
                    [h_sfbdt[bh.loc(sam_axes.value(k)), bh.loc(i), pt_loc, :] \
                        for k in range(sam_axes.size) if sam_axes.value(k) != 'jetht']
                )
                # store hist into numerical values
                _stored = {
                    'edges': h_data_fj.axes[0].edges.tolist(),
                    'h_data': h_data_fj.values(flow=True).tolist(),
                    'h_mc': h_mc_fj.values(flow=True).tolist(),
                    'h_w': np.clip(h_data_fj.values(flow=True) / np.maximum(h_mc_fj.values(flow=True), 1e-20), 0., 2.).tolist(),
                }
                hist_values[f'fj{i}_pt{ptmin}to{ptmax}'] = _stored

        with open(os.path.join(self.outputdir, 'sfbdt_hist.json'), 'w') as fw:
            json.dump(hist_values, fw, indent=4)


    def make_webpage(self):

        _logger.info('[Make webpage]: Making the sfBDT coastline.')

        if not hasattr(self, 'coastline_map') or self.coastline_map is None:
            self.load_pickle('coastline_map')

        # Configure mplhep
        if self.global_cfg.use_helvetica == True or (self.global_cfg.use_helvetica == 'auto' and any(['Helvetica' in font for font in mpl.font_manager.findSystemFonts()])):
            plt.style.use({"font.sans-serif": 'Helvetica'})
        year, lumi = self.global_cfg.year, self.global_cfg.lumi_dict[self.global_cfg.year]

        # Init the web maker
        web = WebMaker(self.job_name)

        # first make the tagger transformation map
        f, ax = plt.subplots(figsize=(10, 10))
        hep.cms.label(data=True, llabel='Preliminary', year=year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % lumi, fontname='sans-serif')
        tmin, tmax = self.global_cfg.tagger.span
        ax.plot(self.signal_hist_info['t_x'], self.signal_hist_info['t_y'], color='royalblue')
        ax.set_xlim(tmin, tmax); ax.set_ylim(0., 1.)
        ax.set_xlabel('Tagger'); ax.set_ylabel('Transformed tagger')
        ax.grid()
        ax.text(0.05, 0.92, 'Tagger transformation map', transform=ax.transAxes)
        plt.savefig(os.path.join(self.webdir, f'tagger_trans.png'))
        plt.savefig(os.path.join(self.webdir, f'tagger_trans.pdf'))
        plt.close()


        # then make signal shape on the original and transformed tagger, and positions of the WPs
        sig_effs = []
        for histname, desc, xlims, xtitle, color in zip(['hist', 'xhist'], ['original', 'transformed'], \
            [(0., 1.), (tmin, tmax)], ['Tagger', 'Transformed tagger'], ['grey', 'royalblue']):
            f, ax = plt.subplots(figsize=(10, 10))
            hep.cms.label(data=True, llabel='Preliminary', year=year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % lumi, fontname='sans-serif')
            h = self.signal_hist_info[histname]
            if h is not None:
                hep.histplot(h.values(), yerr=np.sqrt(h.variances()), bins=h.axes[0].edges, label='Signal', color=color)
            ax.set_xlim(xlims); ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
            ymax = ax.get_ylim()[1]
            for wp, (lo, hi) in self.global_cfg.tagger.wps.items():
                if histname == 'xhist': # do transformation on WP edges
                    lo, hi = self.xtagger_map(lo), self.xtagger_map(hi)
                    sig_effs.append((lo, hi)) # store the bounds of the transformed tagger score, equivlent as the signal efficiencies at the original tagger
                ax.plot([lo, lo], [0., ymax], color='grey', linestyle='dotted') # vertical line to separate WPs
                ax.text((lo + hi)/2, 0.82 * ymax, wp, ha='center', fontweight='bold')
            ax.set_xlabel(xtitle); ax.set_ylabel('Events / bin')
            ax.text(0.05, 0.92, f'Signal distribution on {desc} tagger', transform=ax.transAxes)
            if h is None:
                 ax.text(0.4, 0.5, 'Not valid', transform=ax.transAxes)
            plt.savefig(os.path.join(self.webdir, f'{histname}_signal.png'))
            plt.savefig(os.path.join(self.webdir, f'{histname}_signal.pdf'))
            plt.close()

        web.add_h1("Tagger transformation")
        anatree = self.global_cfg.main_analysis_tree
        if self.provide_tagger_array:
            web.add_text(f"Signal sample: `{anatree.path.replace('$YEAR', str(self.global_cfg.year))}:{anatree.treename}`.")
            web.add_text(f"Applied selection: `{anatree.selection}`. Tagger name/expression: `{anatree.tagger}`\n")
            web.add_text(f"Defined WPs: {self.global_cfg.tagger.wps}\n")
            web.add_text(r"This corresponds to the signal effiency at: {%s}" %
                ', '.join([f'{wp}: [{lo*100:.1f}\%, {hi*100:.1f}\%]' for wp, (lo, hi) in zip(self.global_cfg.tagger.wps.keys(), sig_effs)]))
            web.add_text()
        else:
            web.add_text(f"Provided signal tagger shape in histogram: `{anatree.path.replace('$YEAR', str(self.global_cfg.year))}:{anatree.histname}`.")
            web.add_text()

        web.add_figure(self.webdir, src='tagger_trans.png', title='tagger transformation map')
        for histname, desc in zip(['hist', 'xhist'], ['original', 'transformed']):
            web.add_figure(self.webdir, src=f'{histname}_signal.png', title=f'signal distribution on {desc} tagger and WPs')


        web.add_h1("sfBDT coastline")
        web.add_text("Left to right: coastline for each jet pT range:")
        pt_edges = self.global_cfg.pt_edges + [100000]
        web.add_text(', '.join([f'({ptmin}, {ptmax})' for ptmin, ptmax in zip(pt_edges[:-1], pt_edges[1:])]))
        web.add_text()

        web.add_h2('Transformed tagger - sfBDT 2D distribution')
        for i, (ptmin, ptmax) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
            f, ax = plt.subplots(figsize=(10, 10))
            hep.cms.label(data=True, llabel='Preliminary', year=year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % lumi, fontname='sans-serif')
            arr2d =  self.coastline_map[i]['arr2d']
            arr2d[arr2d == 0.] = np.nan # leave the pixel blank if no entry
            im = ax.imshow(arr2d[:, ::-1].T, norm=mpl.colors.LogNorm(), interpolation='nearest', extent=[0, 1, 0, 1], cmap=plt.cm.jet)
            f.colorbar(im, ax=ax)
            ax.set_xlabel('Transformed tagger'); ax.set_ylabel('sfBDT')

            plt.savefig(os.path.join(self.webdir, f'h2d_{year}_pt{ptmin}to{ptmax}.png'))
            plt.savefig(os.path.join(self.webdir, f'h2d_{year}_pt{ptmin}to{ptmax}.pdf'))
            plt.close()

            web.add_figure(self.webdir, src=f'h2d_{year}_pt{ptmin}to{ptmax}.png', title=f'pT ({ptmin}, {ptmax})')
        
        web.add_h2('Coastlines')

        for i, (ptmin, ptmax) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
            # draw contour as the coastlines
            f, ax = plt.subplots(figsize=(10, 10))
            hep.cms.label(data=True, llabel='Preliminary', year=year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % lumi, fontname='sans-serif')
            step = 1./ self.nbin2d
            x = y = np.arange(step/2, 1. + step/2, step) ## 200 bins correspond to the hist
            Y, X = np.meshgrid(y, x)
            CS = ax.contour(X, Y, self.coastline_map[i]['arr2d_cum_smeared'], levels=self.coastline_map[i]['levels'])
            ax.clabel(CS, inline=0, fontsize=10)
            ax.set_xlabel('Transformed tagger'); ax.set_ylabel('sfBDT')

            plt.savefig(os.path.join(self.webdir, f'coastline_{year}_pt{ptmin}to{ptmax}.png'))
            plt.savefig(os.path.join(self.webdir, f'coastline_{year}_pt{ptmin}to{ptmax}.pdf'))
            plt.close()

            web.add_figure(self.webdir, src=f'coastline_{year}_pt{ptmin}to{ptmax}.png', title=f'pT ({ptmin}, {ptmax})')
        
        web.write_to_file(self.webdir)