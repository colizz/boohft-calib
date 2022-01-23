"""
For step 4: apply the fit to derive SFs, then make plots for the fit.

"""

import numpy as np
import uproot
import uproot3

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cycler import cycler 
mpl.use('Agg')
mpl.rcParams['axes.prop_cycle'] = cycler(color=['blue', 'red', 'green', 'violet', 'darkorange', 'black', 'cyan', 'yellow'])
import mplhep as hep
plt.style.use(hep.style.CMS)

from types import SimpleNamespace
import subprocess
import pickle
import shutil
import glob
import os

from unit import ProcessingUnit, StandaloneMultiThreadedUnit
from utils.web_maker import WebMaker
from utils.plotting import make_shape_unce_plots, make_prepostfit_plots, make_stacked_plots, make_sfbdt_variation_plot, make_fit_summary_plots
from logger import _logger


def runcmd(cmd, shell=True):
    """Run a shell command"""
    p = subprocess.Popen(
        cmd, shell=shell, universal_newlines=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE
    )
    out, _ = p.communicate()
    return (out, p.returncode)


class FitUnit(ProcessingUnit):
    r"""The unit processing wrapper of the second step (calculate the coastline and derive the fit template"""

    def __init__(self, global_cfg, job_name='4_fit', job_name_step1='1_mc_reweight', job_name_step2='2_coastline', 
                 job_name_step3='3_tmpl_writer', fileset=None, **kwargs):
        super().__init__(
            job_name=job_name,
            fileset=fileset,
            processor_cls=None, # no need for coffea processor
            processor_kwargs={'global_cfg': global_cfg},
            **kwargs,
        )
        self.global_cfg = global_cfg
        self.job_name_step1 = job_name_step1
        self.job_name_step2 = job_name_step2
        self.job_name_step3 = job_name_step3
        self.outputdir = os.path.join('output', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name)
        self.outputdir_step1 = os.path.join('output', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name_step1)
        self.outputdir_step2 = os.path.join('output', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name_step2)
        self.outputdir_step3 = os.path.join('output', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name_step3)
        self.webdir = os.path.join('web', self.global_cfg.routine_name + '_' + str(self.global_cfg.year), self.job_name)
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
        if not os.path.exists(self.webdir):
            os.makedirs(self.webdir)
        

    def load_pickle(self, attrname: str):
        if not os.path.isfile(os.path.join(self.outputdir, attrname + '.pickle')):
            _logger.exception('Cannot find ' + os.path.join(self.outputdir, attrname + '.pickle'))
            raise

        with open(os.path.join(self.outputdir, attrname + '.pickle'), 'rb') as f:
            setattr(self, attrname, pickle.load(f))


    def preprocess(self):
        def _check_exists(dirpath, filename, jobname):
            if not os.path.isfile(os.path.join(dirpath, filename)):
                _logger.exception('Cannot find ' + os.path.join(dirpath, filename) + '\n' \
                    + 'Please run step ' + jobname + ' first.')
                raise

        # Load the template first
        _check_exists(self.outputdir_step3, 'tmpl_fitpath.pickle', self.job_name_step3)
        with open(os.path.join(self.outputdir_step3, 'tmpl_fitpath.pickle'), 'rb') as f:
            self.tmpl_fitpath = pickle.load(f)
        
        # Load the fitting and plotting arguments
        self.fit_options = SimpleNamespace( # args for plot maker
            year=self.global_cfg.year,
            lumi=self.global_cfg.lumi_dict[self.global_cfg.year],
            type=self.global_cfg.type, # bb or cc calibration type
            skip_fit=self.global_cfg.skip_fit,
            run_impact_for_central_fit=self.global_cfg.run_impact_for_central_fit,
            use_helvetica=self.global_cfg.use_helvetica,
            color_order=sns.color_palette('cubehelix_r', 3),
            cat_order=['flvL', 'flvC', 'flvB'] if self.global_cfg.type=='bb' else ['flvL', 'flvB', 'flvC'],
            unce_list=self.global_cfg.unce_list,
            xlabel=r'$log(m_{SV1,d_{xy}sig\,max}\; /GeV)$',
        )

    def postprocess(self):

        _logger.info("[Postprocess]: Launch the fit in the threaded workflow.")

        # 1. Setup CMSSW environment
        out, ret = runcmd("bash cmssw/env_setup.sh")
        if ret != 0:
            _logger.exception("Error running cmssw setup:\n\n" + out)
            raise
        
        # 2. Launch the fit then make plots
        fit_handler = StandaloneMultiThreadedUnit(workers=self.workers, use_unordered_mapping=True)

        def is_central_sfbdt(path):
            r"""find the central BDT point"""
            bdtdirs = os.listdir(os.path.join(path, '..'))
            assert len(bdtdirs) % 2 == 1
            c_bdt = sorted(bdtdirs)[int((len(bdtdirs) - 1)/2)]
            return os.path.basename(os.path.normpath(path)) == c_bdt

        def get_fit_arguments():
            # Launch in three fit mode:
            n_fit = 0
            args_collection = []
            for mode, readflag in zip(['main', 'sfbdt_rwgt', 'fit_var_rwgt'], \
                [self.global_cfg.do_main_fit, self.global_cfg.do_sfbdt_rwgt_fit, self.global_cfg.do_fit_var_rwgt_fit]):
                if readflag: # ok, will run this fit scheme
                    non_central_fit = []
                    for path in self.tmpl_fitpath:
                        # workpath is under "4_fit/{mode}" while input card path is under "3_tmpl_writer/cards"
                        workpath = path.replace(self.job_name_step3 + '/cards', self.job_name + f'/{mode}')
                        is_central = is_central_sfbdt(path)
                        if is_central:
                            # launch central BDT first as it may take longer time
                            # the central BDT cut is the nominal fit point. Run more fit utilities for these cases
                            args_collection.append((path, workpath, self.fit_options, is_central, mode,))
                            n_fit += 1
                            if self.global_cfg.test_n_fit != -1 and n_fit >= self.global_cfg.test_n_fit:
                                return args_collection
                        else:
                            non_central_fit.append((path, workpath))
                    if not self.global_cfg.run_central_fit_only:
                        for path, workpath in non_central_fit:
                            # launch non-central points after all central ones are submitted
                            args_collection.append((path, workpath, self.fit_options, is_central, mode,))
                            if self.global_cfg.test_n_fit != -1 and n_fit >= self.global_cfg.test_n_fit:
                                return args_collection
            return args_collection

        # run all fits concurrently
        for args in get_fit_arguments():
            fit_handler.book(args)
        result = fit_handler.run(concurrent_fit_unit)

        # summarize when all jobs are done
        failed_dirpath = []
        for path, ret in result:
            if ret != 0:
                failed_dirpath.append(path)
        if len(failed_dirpath):
            _logger.error("Summary of all failed fit point: [\n   " + '\n   '.join(failed_dirpath) + '\n]')


    def make_webpage(self):
        r"""Summarize all fit results and plots in one webpage.
        Inherit from the web_mode making code: https://github.com/colizz/ParticleNet-CCTagCalib/blob/main/make_html.py
        """

        _logger.info('[Make webpage]: Collecting all derived SF results and plots in one webpage.')
        pt_edges = self.global_cfg.pt_edges + [100000]
        pt_edge_pairs = [(ptmin, ptmax) for ptmin, ptmax in zip(pt_edges[:-1], pt_edges[1:])]
        pt_list = [f'pt{ptmin}to{ptmax}' for ptmin, ptmax in zip(pt_edges[:-1], pt_edges[1:])]
        wps = self.global_cfg.tagger.wps
        pjoin = os.path.join
        pt_title = lambda ptmin, ptmax: '[{ptmin}, {ptmax})'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+inf')
        for wp in wps:
            if wp not in self.global_cfg.default_wp_name_map:
                self.global_cfg.default_wp_name_map[wp] = wp
        if self.global_cfg.type == 'bb':
            sf_list = ['B', 'C', 'L']
            sf_title = {'B':'bb-tagging SF (`SF_flvB`)', 'C':'cc-mistagging SF (`SF_flvC`)', 'L':'light-mistagging SF (`SF_flvL`)'}
        elif self.global_cfg.type == 'cc':
            sf_list = ['C', 'B', 'L']
            sf_title = {'C':'cc-tagging SF (`SF_flvC`)', 'B':'bb-mistagging SF (`SF_flvB`)', 'L':'light-mistagging SF (`SF_flvL`)'}
        
        def get_dir(mode, wp, pt, bdt, return_list=False):
            r"""Read the list of dir matches the given bdt and pt (wildcard supported)
            """
            # note that self.outputdir stores the fit results from postprocessing step
            outdir_list = glob.glob(pjoin(self.outputdir, mode, wp, pt, bdt))
            if len(outdir_list) == 0:
                raise RuntimeError('Directory does not exist: ' + pjoin(self.outputdir, mode, wp, pt, bdt))
            elif len(outdir_list) == 1 and not return_list:
                return outdir_list[0]
            else:
                return outdir_list

        def get_central_sfbdt(mode, wp, pt, return_idx=False):
            bdtdirs = [os.path.basename(d) for d in get_dir(mode, wp, pt, '*', return_list=True)]
            assert len(bdtdirs) % 2 == 1
            c_idx = int((len(bdtdirs) - 1)/2)
            c_bdt = sorted(bdtdirs)[c_idx]
            return c_bdt if not return_idx else c_idx

        def read_sf_from_log(flist, sf):
            r"""Read the SFs from each of the log file in the given list. Return the list of center, errl, errh values
            """
            if isinstance(flist, str):
                out = [open(flist).readlines()]
            elif isinstance(flist, list):
                out = []
                for f in flist:
                    out.append(open(f).readlines())
            
            center, errl, errh = [], [], []
            for content in out:
                for l in content:
                    l = l.split()
                    if len(l)>0 and l[0]==f'SF_flv{sf}':
                        center.append(float(l[2]))
                        errl.append(float(l[3].split('/')[0]))
                        errh.append(float(l[3].split('/')[1]))
                        break
                else:
                    center.append(np.nan)
                    errl.append(np.nan)
                    errh.append(np.nan)
            return np.array(center), np.array(errl), np.array(errh)

        def merge_sfbdt_variation_margins(web_helper, mode, sf):
            web_helper.add_h2(f'{sf_title[sf]} (after external correction from sfBDT variation)')
            web_helper.add_text('|       | ' + ' | '.join([f'pT {pt_title(ptmin, ptmax)}' for ptmin, ptmax in pt_edge_pairs]) + ' |')
            web_helper.add_text('| :---: '*(len(pt_list)+1) + '|')
            for wp in wps: # for each rows
                center_ptlist, errl_ptlist, errh_ptlist, maxdist_ptlist = [], [], [], []
                c_errl_ptlist, c_errh_ptlist = [], []
                err_fvr_ptlist = []
                for pt in pt_list: # for each column
                    # get all logs for sfBDT variation and sorted by sfBDT order
                    loglist = sorted([pjoin(d, 'fit.log') for d in get_dir(mode, wp, pt, 'bdt*', return_list=True)])
                    center, _, _ = read_sf_from_log(loglist, sf=sf)
                    center = center[center != np.nan]
                    c_center, c_errl, c_errh = read_sf_from_log([pjoin(get_dir(mode, wp, pt, get_central_sfbdt(mode, wp, pt)), 'fit.log')], sf=sf)
                    c_center, c_errl, c_errh = c_center[0], c_errl[0], c_errh[0]
                    import math
                    if not math.isnan(c_center):
                        maxdist = np.max(np.abs(center - c_center))
                        center_ptlist.append(c_center)
                        c_errl_ptlist.append(c_errl)
                        c_errh_ptlist.append(c_errh)
                        maxdist_ptlist.append(maxdist)
                        errl_ptlist.append(-np.sqrt(c_errl**2 + maxdist**2))
                        errh_ptlist.append(np.sqrt(c_errh**2 + maxdist**2))
                    else:
                        center_ptlist.append(np.nan); c_errl_ptlist.append(np.nan); c_errh_ptlist.append(np.nan); maxdist_ptlist.append(np.nan)
                        err_fvr_ptlist.append(np.nan); errl_ptlist.append(np.nan); errh_ptlist.append(np.nan)
                web_helper.add_text(f'| **{self.global_cfg.default_wp_name_map[wp]}** WP | ' + \
                                ' | '.join([f'**{c:.3f}** ([{el:+.3f}/{eh:+.3f}]<sub>orig</sub> [±{md:.3f}]<sub>max d</sub>)</br>[{eltot:+.3f}/{ehtot:+.3f}]<sub>tot</sub>' \
                                    for c, el, eh, md, eltot, ehtot in zip(center_ptlist, c_errl_ptlist, c_errh_ptlist, maxdist_ptlist, errl_ptlist, errh_ptlist)]) + \
                                ' |')

        def merge_sfbdt_variation_smearing(web_helper, mode, sf):
            web_helper.add_h2(f'{sf_title[sf]} (after smearing over all sfBDT variation points)')
            web_helper.add_text('|       | ' + ' | '.join([f'pT {pt_title(ptmin, ptmax)}' for ptmin, ptmax in pt_edge_pairs]) + ' |')
            web_helper.add_text('| :---: '*(len(pt_list)+1) + '|')

            def to_float_fix_nan(x_in: list):
                x_out = []
                n = len(x_in)
                for i, x in enumerate(x_in):
                    if x != np.nan:
                        x_out.append(x)
                    else: # interpolate NaNs
                        for p in range(1, 10):
                            if x_in[(i+p) % n] != np.nan and x_in[(i-p) % n] != np.nan:
                                x_out.append(0.5 * (x_in[(i+p) % n] + x_in[(i-p) % n]))
                                break
                        else:
                            _logger.critical('The fit has too many NaNs... Lucky day for you.')
                            raise
                return x_out

            from scipy.interpolate import interp1d
            from scipy.stats import norm

            s_centers_all, s_errls_all, s_errhs_all = [], [], [] # collect all results as return values
            for wp in wps: # for each rows
                s_centers, s_errls, s_errhs = [], [], []
                for pt in pt_list: # for each column 
                    # get all logs for sfBDT variation and sorted by sfBDT order
                    loglist = sorted([pjoin(d, 'fit.log') for d in get_dir(mode, wp, pt, 'bdt*', return_list=True)])
                    center, errl, errh = map(to_float_fix_nan, read_sf_from_log(loglist, sf=sf))
                    # c_center, c_errl, c_errh = map(to_float_fix_nan, read_sf_from_log([pjoin(get_dir(mode, wp, pt, get_central_sfbdt(mode, wp, pt)), 'fit.log')], sf=sf))

                    nbdt = len(center)
                    # average quantiles over all fit points
                    func_quantile = lambda x: sum([norm.cdf(val) for val in list((x-center>=0) * (x-center)/errh + (x-center<0) * (-x+center)/errl)]) / nbdt
                    x = np.linspace(0., 2., 100)
                    y = np.array([func_quantile(i) for i in x])
                    finv = interp1d(y, x)
                    s_center, s_lo, s_hi = finv(norm.cdf(0)), finv(norm.cdf(-1)), finv(norm.cdf(1))
                    s_errl, s_errh = s_lo - s_center, s_hi - s_center
                    s_centers.append(s_center); s_errls.append(s_errl); s_errhs.append(s_errh)
                web_helper.add_text(
                        f'| **{self.global_cfg.default_wp_name_map[wp]}** WP | ' \
                        + ' | '.join([f'**+{c:.3f}** [{el:.3f}/+{eh:.3f}]' for c, el, eh in zip(s_centers, s_errls, s_errhs)]) + ' |'
                    )
                s_centers_all.append(s_centers); s_errls_all.append(s_errls); s_errhs_all.append(s_errhs)

            return np.array(s_centers_all), np.array(s_errls_all), np.array(s_errhs_all)


        ## Initialize a webpage instance
        web = WebMaker(self.job_name)

        center_fin, errl_fin, errh_fin = {}, {}, {} # reserve SF results used for write outer webpage
        ## Writing three webpages each for one fit scheme
        self.modes = ['main', 'sfbdt_rwgt', 'fit_var_rwgt']
        self.mode_names = ['main scheme', 'sfBDT reweighting scheme', 'Fit variable reweighting scheme']
        for mode, mode_name in zip(self.modes, self.mode_names):
            _logger.debug(f'Origanizing webpage for mode: {mode}')

            web_mode = WebMaker(self.job_name + '/' + mode)
            webdir_mode = os.path.join(self.webdir, mode)
            if not os.path.exists(webdir_mode):
                os.makedirs(webdir_mode)
            web_mode.add_h1(f"Central `sfBDT` coastline result (mode: {mode_name})")

            ## fit result
            for sf in sf_list:
                web_mode.add_h2(sf_title[sf])
                web_mode.add_text('|       | ' + ' | '.join([f'pT {pt_title(ptmin, ptmax)}' for ptmin, ptmax in pt_edge_pairs]) + ' |')
                web_mode.add_text('| :---: '*(len(pt_list)+1) + '|')
                for wp in wps:
                    loglist = [pjoin(get_dir(mode, wp, pt, get_central_sfbdt(mode, wp, pt)), 'fit.log') for pt in pt_list]
                    center, errl, errh = read_sf_from_log(loglist, sf=sf) ## sf set to the correct sf!
                    web_mode.add_text(
                        f'| **{self.global_cfg.default_wp_name_map[wp]}** WP | ' \
                        + ' | '.join([f'**{c:.3f}** [{el:+.3f}/{eh:+.3f}]' for c, el, eh in zip(center, errl, errh)]) + ' |'
                    )
                    if sum(center == np.nan) > 0:
                        _logger.error(f'multifit failed... ', wp, 'central SFs: ', center)

                ## include an extra table for merging all sfBDT variation results
                if sf == sf_list[0] or self.global_cfg.show_sfbdt_variation_all_flavour:
                    # merge_sfbdt_variation_margins(web_mode, mode, sf) ## previously adopted, sometimes not stable
                    center_fin[mode], errl_fin[mode], errh_fin[mode] = merge_sfbdt_variation_smearing(web_mode, mode, sf) # we will take these SFs to the summary


            if not self.global_cfg.show_fit_number_only:
                # gather the plots to the webdir folder
                def collect_figure_to_web(web_mode, orig_path, suffix, title, type='figure'):
                    fname = orig_path.rsplit('/', 1)[-1].split('.')[0] + suffix + '.' + orig_path.rsplit('/', 1)[-1].split('.')[1]
                    if os.path.isfile(orig_path):
                        shutil.copy2(orig_path, os.path.join(webdir_mode, fname)) # copy and rename to append the suffix
                    else:
                        _logger.error('File ' + orig_path + ' does not exists. Do some fit points fail?')
                    if type == 'figure':
                        web_mode.add_figure(webdir_mode, fname, title)
                    elif type == 'pdf':
                        web_mode.add_pdf(webdir_mode, fname, title, width=700, height=500)

                for wp in wps:
                    ## gather the pre and post-fit stack plots
                    for prepost in ['prefit', 'postfit']:
                        if prepost=='prefit':
                            web_mode.add_h2(f'Pre-fit stacked plot (**{self.global_cfg.default_wp_name_map[wp]}** WP)')
                        elif prepost=='postfit':
                            web_mode.add_h2(f'Post-fit stacked plot (**{self.global_cfg.default_wp_name_map[wp]}** WP)')
                        for wpcat in ['pass', 'fail']:
                            # left->right: pT increase
                            web_mode.add_h3(wpcat)
                            web_mode.add_text(f'In the order of : pT in ' + ', '.join([pt_title(ptmin, ptmax) for ptmin, ptmax in pt_edge_pairs]) + '\n\n')
                            for pt, (ptmin, ptmax) in zip(pt_list, pt_edge_pairs):
                                c_bdt = get_central_sfbdt(mode, wp, pt)
                                collect_figure_to_web(web_mode, orig_path=pjoin(get_dir(mode, wp, pt, c_bdt), f'stack_{prepost}_{wpcat}.png'),
                                                    suffix=f'__{wp}__{pt}__{c_bdt}', title=f'{pt_title(ptmin, ptmax)}, {wpcat}'
                                                    )

                    ## gather the pre+postfit template
                    web_mode.add_h2(f'Pre/post-fit template (**{self.global_cfg.default_wp_name_map[wp]}** WP)')
                    for wpcat in ['pass', 'fail']:
                        web_mode.add_h3(wpcat)
                        web_mode.add_text(f'In the order of : pT in ' + ', '.join([pt_title(ptmin, ptmax) for ptmin, ptmax in pt_edge_pairs]) + '\n\n')
                        for pt, (ptmin, ptmax) in zip(pt_list, pt_edge_pairs):
                            c_bdt = get_central_sfbdt(mode, wp, pt)
                            collect_figure_to_web(web_mode, orig_path=pjoin(get_dir(mode, wp, pt, c_bdt), f'prepostfit_{wpcat}.png'),
                                                suffix=f'__{wp}__{pt}__{c_bdt}', title=f'{pt_title(ptmin, ptmax)}, {wpcat}'
                                                )

                    ## gather the uncertainty comparison plots
                    web_mode.add_h2(f'Shape uncetainty variations (**{self.global_cfg.default_wp_name_map[wp]}** WP)')
                    for unce_type in self.global_cfg.unce_list:
                        web_mode.add_h3(f'>>> {unce_type}')
                        for wpcat in ['pass', 'fail']:
                            web_mode.add_h3(wpcat)
                            web_mode.add_text('In the order of : pT in ' + ', '.join([pt_title(ptmin, ptmax) for ptmin, ptmax in pt_edge_pairs]) + '\n\n')
                            for pt, (ptmin, ptmax) in zip(pt_list, pt_edge_pairs):
                                c_bdt = get_central_sfbdt(mode, wp, pt)
                                collect_figure_to_web(web_mode, orig_path=pjoin(get_dir(mode, wp, pt, c_bdt), f'unce_comp_{unce_type}_{wpcat}.png'),
                                                    suffix=f'__{wp}__{pt}__{c_bdt}', title=f'{pt_title(ptmin, ptmax)}, {wpcat}'
                                                    )

                    ## impact plot
                    web_mode.add_h2(f'Impacts plot (**{self.global_cfg.default_wp_name_map[wp]}** WP)')
                    web_mode.add_text(f'In the order of : pT in ' + ', '.join([pt_title(ptmin, ptmax) for ptmin, ptmax in pt_edge_pairs]) + '\n\n')
                    for pt, (ptmin, ptmax) in zip(pt_list, pt_edge_pairs):
                        c_bdt = get_central_sfbdt(mode, wp, pt)
                        collect_figure_to_web(web_mode, orig_path=pjoin(get_dir(mode, wp, pt, c_bdt), 'impacts.pdf'),
                                            suffix=f'__{wp}__{pt}__{c_bdt}', title=f'{pt_title(ptmin, ptmax)}', type='pdf'
                                            )

                web_mode.add_text('------------------\n')

            if self.global_cfg.show_unce_breakdown:
                web_mode.add_h1(f'{self.global_cfg.type}-tagging SF uncetainty breakdown for syst. and stat.')
                for wp in wps:
                    web_mode.add_h2(f'**{self.global_cfg.default_wp_name_map[wp]}** WP:')
                    web_mode.add_text('Left to right: pT in ' + ', '.join([pt_title(ptmin, ptmax) for ptmin, ptmax in pt_edge_pairs]) + '\n\n')
                    for pt, (ptmin, ptmax) in zip(pt_list, pt_edge_pairs):
                        c_bdt = get_central_sfbdt(mode, wp, pt)
                        collect_figure_to_web(web_mode, orig_path=pjoin(get_dir(mode, wp, pt, c_bdt), 'unce_breakdown.png'),
                                            suffix=f'__{wp}__{pt}__{c_bdt}', title=f'{pt_title(ptmin, ptmax)}'
                                            )
            
            # make SF plots for varied sfBDT cut
            if self.global_cfg.show_sfbdt_variation:
                web_mode.add_h1(f'{self.global_cfg.type}-tagging SFs for the variation of sfBDT coastlines')
                for wp in wps:
                    web_mode.add_h2(f'**{self.global_cfg.default_wp_name_map[wp]}** WP:')
                    web_mode.add_text('Left to right: pT in ' + ', '.join([pt_title(ptmin, ptmax) for ptmin, ptmax in pt_edge_pairs]) + '\n\n')
                    sf_draw = sf_list[:1] if not self.global_cfg.show_sfbdt_variation_all_flavour else sf_list
                    for sf in sf_draw:
                        for pt, (ptmin, ptmax) in zip(pt_list, pt_edge_pairs):
                            plot_name = f'sfbdtvary_{sf}__{wp}__{pt}'
                            plot_text = (self.global_cfg.default_wp_name_map[wp], r'$p_{T}$: '+f'{pt_title(ptmin, ptmax)}'+' GeV')
                            title = pt_title(ptmin, ptmax)
                            if not self.global_cfg.show_sfbdt_variation_norun:
                                _logger.debug(f'Producing sfBDT variation plots for {wp}, ({ptmin}, {ptmax})')
    
                                c_idx = get_central_sfbdt(mode, wp, pt, return_idx=True)
                                bdtdirs = sorted([os.path.basename(d) for d in get_dir(mode, wp, pt, 'bdt*', return_list=True)]); assert len(bdtdirs) % 2 == 1
                                # read SF results via the sfBDT list
                                loglist = [os.path.join(get_dir(mode, wp, pt, bdt), 'fit.log') for bdt in bdtdirs]
                                center, errl, errh = read_sf_from_log(loglist, sf=sf)
                                # make plots
                                make_sfbdt_variation_plot(center, errl, errh, c_idx, sf, webdir_mode, self.fit_options, plot_text, plot_name)
                            web_mode.add_figure(webdir_mode, plot_name + '.png', title, width=600, height=300)
                        web_mode.add_text()

            web_mode.write_to_file(webdir_mode)

            # Make the summary plot for SFs in this mode
            if mode in center_fin.keys(): # we have stored the eventual SFs for this mode during the processing
                _logger.debug(f'Making summary fit plots for mode: {mode}')
                plot_xticklabels = ['[{ptmin}, {ptmax})'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else r'$\infty$') for ptmin, ptmax in pt_edge_pairs]
                plot_ylabel = f'SF (flv{sf_list[0]})'
                plot_legends = list(wps.keys())
                plot_text = mode_name
                plot_name = f'sf_summary_{sf_list[0]}_{mode}'
                make_fit_summary_plots(center_fin[mode], errl_fin[mode], errh_fin[mode], self.webdir, self.fit_options,
                    plot_xticklabels, plot_ylabel, plot_legends, plot_text, plot_name
                )
            else:
                _logger.critical('Somehow the fit results for each scheme cannot be found. This should not happen.')

        # end of mode for-loop
        
        # Now let's write the outer webpage
        # Summarize all fit results and make plots
        if mode in center_fin.keys():
            plot_text = 'Final set'
            plot_name = f'sf_summary_{sf_list[0]}_comb'
            center_comb = center_fin['main']
            maxdist = np.maximum(*tuple([np.abs(center_fin[m] - center_fin['main']) for m in ['sfbdt_rwgt', 'fit_var_rwgt']]))
            errl_comb = -np.hypot(errl_fin['main'], maxdist)
            errh_comb = np.hypot(errh_fin['main'], maxdist)
            make_fit_summary_plots(center_comb, errl_comb, errh_comb, self.webdir, self.fit_options,
                plot_xticklabels, plot_ylabel, plot_legends, plot_text, plot_name
            )
        # Write contents on the outer webpage
        web.add_h1('Final SF results')
        web.add_h2(sf_title[sf_list[0]])
        web.add_text('|       | ' + ' | '.join([f'pT {pt_title(ptmin, ptmax)}' for ptmin, ptmax in pt_edge_pairs]) + ' |')
        web.add_text('| :---: '*(len(pt_list)+1) + '|')
        for i, wp in enumerate(wps):
            center, errl, errh = center_comb[i], errl_comb[i], errh_comb[i]
            web.add_text(
                f'| **{self.global_cfg.default_wp_name_map[wp]}** WP | ' \
                + ' | '.join([f'**{c:.3f}** [{el:+.3f}/{eh:+.3f}]' for c, el, eh in zip(center, errl, errh)]) + ' |'
            )
        web.add_text()
        web.add_figure(self.webdir, src=f'sf_summary_{sf_list[0]}_comb.png', title='Final set')
        web.add_text()

        # write SF summary in latex format
        latex_pt_lines = ' '*4 + ' '.join(['& $[{ptmin},~{ptmax})$'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '\\infty') for ptmin, ptmax in pt_edge_pairs])
        latex_sf_content = []
        for i, wp in enumerate(wps):
            latex_sf_line = []
            latex_sf_line.append(f'{wp}: $[{wps[wp][0]},~{wps[wp][1]})$')
            for j in range(len(pt_list)):
                latex_sf_line.append(r'${%.3f}_{%+.3f}^{%+.3f}$' % (center_comb[i][j], errl_comb[i][j], errh_comb[i][j]))
            latex_sf_content.append(' '*4 + ' & '.join(latex_sf_line) + r' \\')
        latex_sf_content = '\n'.join(latex_sf_content)

        latex_str_template = r'''\begin{table}[!h]
\centering
\caption{\label{tab:hrt_calib_results}Summary of the %s-tagging SF.}
%% \resizebox*{1\textwidth}{!}{
  \begin{tabular}{l%s} \hline
    & \multicolumn{%d}{c}{\pt\ range in GeV:} \\
%s \\ \hline\hline
%s
    \hline\hline
  \end{tabular}
%% }
\end{table}
'''
        latex_str = latex_str_template % (
            self.global_cfg.type, # fill the caption
            '|c' * center_comb.shape[1], # format the table
            center_comb.shape[1], # format the table
            latex_pt_lines,
            latex_sf_content
        )
        web.add_text('Copy the LaTeX format below for easy documentation.')
        web.add_text('Detailed fit results and plots can be found in the **following links**.\n\n')
        web.end_this_md_block() # pure text must not be put under the zero-md tag otherwise the md symbols are interpreted..
        web.add_textarea(latex_str, width=1200, height=250)
        web.begin_new_md_block()

        # conclude results for each of the three modes
        web.add_text('\n---------------')
        web.add_h1('Separate results from the three modes')
        for mode, mode_name in zip(self.modes, self.mode_names):
            web.add_figure(self.webdir, src=f'sf_summary_{sf_list[0]}_{mode}.png', title=f'SF set for mode: {mode}')
        for mode, mode_name in zip(self.modes, self.mode_names):
            web.add_h2(sf_title[sf_list[0]] + f' for [{mode_name}]({mode})')
            web.add_text('|       | ' + ' | '.join([f'pT {pt_title(ptmin, ptmax)}' for ptmin, ptmax in pt_edge_pairs]) + ' |')
            web.add_text('| :---: '*(len(pt_list)+1) + '|')
            for i, wp in enumerate(wps):
                center, errl, errh = center_fin[mode][i], errl_fin[mode][i], errh_fin[mode][i]
                web.add_text(
                    f'| **{self.global_cfg.default_wp_name_map[wp]}** WP | ' \
                    + ' | '.join([f'**{c:.3f}** [{el:+.3f}/{eh:+.3f}]' for c, el, eh in zip(center, errl, errh)]) + ' |'
                )

        web.write_to_file(self.webdir)
        

def concurrent_fit_unit(arg):
    r"""A unit function to run a single fit point for the given workdir.
    This function is launched in the multiprocessing pool.
    Inherit from the fit code from https://github.com/colizz/ParticleNet-CCTagCalib/blob/main/cmssw/fit.py
      and plotting code from https://github.com/colizz/ParticleNet-CCTagCalib/blob/main/make_plots.py

    Sealed arguments:
        workdir: the base directory path for the fit point
        args: args used to make plots
        is_central: is the central fit point
    """
    inputdir, workdir, args, is_central, mode = arg
    # OpenBLAS' multithreading may confict with the main program's multithreading
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # 1. Launch the fit
    if not args.skip_fit:
        _logger.debug("Run fit point " + workdir)
        if is_central and args.run_impact_for_central_fit:
            out, ret = runcmd(f"bash cmssw/launch_fit.sh {inputdir} {workdir} --type={args.type} --mode={mode} --run-impact --run-unce-breakdown")
        else:
            out, ret = runcmd(f"bash cmssw/launch_fit.sh {inputdir} {workdir} --type={args.type} --mode={mode}")
        if ret != 0:
            _logger.error("Error running the fit point: " + workdir + "\n" + \
                "See the following output (from last few lines):\n\n" + '\n'.join(out.splitlines()[-20:]))
            return (workdir, ret)

    # 2. Make plots if fit succeeds and is the central fit point
    if is_central:
        if args.use_helvetica == True or (args.use_helvetica == 'auto' and any(['Helvetica' in font for font in mpl.font_manager.findSystemFonts()])):
            plt.style.use({"font.sans-serif": 'Helvetica'})

        make_stacked_plots(inputdir, workdir, args, save_plots=True)
        make_prepostfit_plots(inputdir, workdir, args, save_plots=True)

        for unce_type in args.unce_list:
            make_shape_unce_plots(inputdir, workdir, args, unce_type=unce_type, save_plots=True)

    return (workdir, 0)

