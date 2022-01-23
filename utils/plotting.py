import numpy as np
import uproot

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cycler import cycler 
mpl.use('Agg')
mpl.rcParams['axes.prop_cycle'] = cycler(color=['blue', 'red', 'green', 'violet', 'darkorange', 'black', 'cyan', 'yellow'])
import mplhep as hep
plt.style.use(hep.style.CMS)

import os
from logger import _logger

def set_sns_color(*args):
    sns.palplot(sns.color_palette(*args))
    sns.set_palette(*args)


def make_stacked_plots(inputdir, workdir, args, plot_unce=True, save_plots=True):
    r"""Make the stacked histograms for both pre-fit and post-fit based on the fitDiagnosticsTest.root
    Arguments:
        inputdir: Directory for input cards
        workdir: Directory to fitDiagnosticsTest.root
        plot_unce: If or not plot the MC uncertainty in the upper & lower panel. Default: True
    """

    ## Get the bin info based on workdir
    edges = uproot.open(f'{inputdir}/inputs_pass.root:data_obs').axis().edges()
    xmin, xmax, nbin = min(edges), max(edges), len(edges)
    # _logger.debug(f'Making stacked plots: {workdir}')
    
    ## All information read from fitDiagnosticsTest.root
    fit = uproot.open(f'{workdir}/fitDiagnosticsTest.root')
    for rootdir, title in zip(['shapes_prefit', 'shapes_fit_s'], ['prefit', 'postfit']):
        for b in ['pass', 'fail']:
            set_sns_color(args.color_order)
            f = plt.figure(figsize=(12, 12))
            gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05) 
            
            ## Upper histogram panel
            ax = f.add_subplot(gs[0])
            hep.cms.label(data=True, llabel='Preliminary', year=args.year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % args.lumi, fontname='sans-serif')
            ax.set_xlim(xmin, xmax); ax.set_xticklabels([]); 
            ax.set_ylabel('Events / bin', ha='right', y=1.0)

            content = [fit[f'{rootdir}/{b}/{cat}'].values() for cat in args.cat_order]
            hep.histplot(content, bins=edges, label=[f'MC ({cat})' for cat in args.cat_order], histtype='fill', edgecolor='k', linewidth=1, stack=True) ## draw MC
            bkgtot, bkgtot_err = fit[f'{rootdir}/{b}/total'].values(), np.sqrt(fit[f'{rootdir}/{b}/total'].variances())
            if plot_unce:
                ax.fill_between(edges, (bkgtot-bkgtot_err).tolist()+[0], (bkgtot+bkgtot_err).tolist()+[0], label='BKG total unce.', 
                                step='post', hatch='\\\\', edgecolor='dimgrey', facecolor='none', linewidth=0
                                ) ## draw bkg unce.
            data, data_errh, data_errl = fit[f'{rootdir}/{b}/data'].values(1), fit[f'{rootdir}/{b}/data'].errors('high')[1], fit[f'{rootdir}/{b}/data'].errors('low')[1]
            hep.histplot(data, yerr=(data_errl, data_errh), bins=edges, label='Data', histtype='errorbar', color='k', markersize=15, elinewidth=1.5) ## draw data
            ax.set_ylim(0, ax.get_ylim()[1])
            if plot_unce:
                ax.set_ylim(0, 1.8*max(data))
            ax.legend()

            ## Ratio panel
            ax1 = f.add_subplot(gs[1]); ax1.set_xlim(xmin, xmax); ax1.set_ylim(0.001, 1.999)
            ax1.set_xlabel(args.xlabel, ha='right', x=1.0); ax1.set_ylabel('Data / MC', ha='center')
            ax1.plot([xmin, xmax], [1, 1], 'k'); ax1.plot([xmin, xmax], [0.5, 0.5], 'k:'); ax1.plot([xmin, xmax], [1.5, 1.5], 'k:')

            if plot_unce:
                ax1.fill_between(edges, ((bkgtot-bkgtot_err)/bkgtot).tolist()+[0], ((bkgtot+bkgtot_err)/bkgtot).tolist()+[0],
                                    step='post', hatch='\\\\', edgecolor='dimgrey', facecolor='none', linewidth=0
                                    ) ## draw bkg unce.
            hep.histplot(data/bkgtot, yerr=(data_errl/bkgtot, data_errh/bkgtot), bins=edges, histtype='errorbar', color='k', markersize=15, elinewidth=1)

            plot_unce_suf = '' if plot_unce else 'noUnce'
            
            if save_plots:
                plt.savefig(f'{workdir}/stack_{title}_{b}{plot_unce_suf}.png')
                plt.savefig(f'{workdir}/stack_{title}_{b}{plot_unce_suf}.pdf')
                plt.close()


def make_prepostfit_plots(inputdir, workdir, args, save_plots=True):
    r"""Make the prefit and postfit shape in one plot"""

    edges = uproot.open(f'{inputdir}/inputs_pass.root:data_obs').axis().edges()
    xmin, xmax, nbin = min(edges), max(edges), len(edges)
    # _logger.debug(f'Making pre/post fit plots: {workdir}')

    # All information read from fitDiagnosticsTest.root
    fit = uproot.open(f'{workdir}/fitDiagnosticsTest.root')
    for b in ['pass', 'fail']:
        f, ax = plt.subplots(figsize=(12, 12))
        hep.cms.label(data=True, llabel='Preliminary', year=args.year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % args.lumi, fontname='sans-serif')
        # fill in data
        data, data_errh, data_errl = fit[f'shapes_prefit/{b}/data'].values(1), fit[f'shapes_prefit/{b}/data'].errors('high')[1], fit[f'shapes_prefit/{b}/data'].errors('low')[1]
        hep.histplot(data, yerr=(data_errl, data_errh), bins=edges, label='Data', histtype='errorbar', color='k', markersize=15, elinewidth=1.5) ## draw data
        # create custom label handles
        from matplotlib.lines import Line2D
        legend_eles = [Line2D([0], [0], marker='o', markersize=10, color='w', markerfacecolor='k', label='Data')]
        legend_eles_style = []
        # draw prefit and postfit in one plot
        for rootdir, title, linestyle in zip(['shapes_prefit', 'shapes_fit_s'], ['prefit', 'postfit'], [':', '-']):
            for icat, (cat, color) in enumerate(zip(['total'] + args.cat_order[::-1], ['black', 'blue', 'red', 'green'])):
                content, yerror = fit[f'{rootdir}/{b}/{cat}'].values(), np.sqrt(fit[f'{rootdir}/{b}/{cat}'].variances())
                hp = hep.histplot(content, yerr=yerror, bins=edges, label=f'{title} ({cat})', color=color, linestyle=linestyle)
                hp[0][1][2][0].set_linestyle(linestyle) # also apply the linestyle to errorbar (set to LineCollection from ErrorbarContainer)
                if rootdir=='shapes_prefit':
                    legend_eles.append(Line2D([0], [0], color=color, label=f'MC ({cat})'))
            legend_eles_style.append(Line2D([0], [0], color='k', linestyle=linestyle, label=title))
        ax.set_xlim(xmin, xmax); ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
        ax.set_xlabel(args.xlabel, ha='right', x=1.0); ax.set_ylabel('Events / bin', ha='right', y=1.0)
        # make legends
        ax_style = plt.legend(handles=legend_eles_style, loc='upper right')
        plt.legend(handles=legend_eles, loc='upper left')
        plt.gca().add_artist(ax_style) # add the second legend

        if save_plots:
            plt.savefig(f'{workdir}/prepostfit_{b}.png')
            plt.savefig(f'{workdir}/prepostfit_{b}.pdf')
            plt.close()


def make_shape_unce_plots(inputdir, workdir, args, unce_type=None, save_plots=True, norm_unce=False):
    r"""Make the shape comparison and/or the stacked histograms for a specific type of shape uncertainty based on the fitDiagnosticsTest.root
    
    Arguments:
        inputdir: Directory for input cards
        workdir: Directory to fitDiagnosticsTest.root
        unce_type: Name of shape uncertainty (w/o Up or Down) to plot.
        save_plots: If or not store the shape comparison plot. Default: True
        norm_unce: Normalize the up/down uncertainty to nominal. Default: False
    """

    edges = uproot.open(f'{inputdir}/inputs_pass.root:data_obs').axis().edges()
    xmin, xmax, nbin = min(edges), max(edges), len(edges)
    # _logger.debug(f'Making shape uncertainty plots: {workdir}')

    # curves for unce
    for b in ['pass', 'fail']:
        content = [uproot.open(f'{inputdir}/inputs_{b}.root:{cat}').values() for cat in args.cat_order[::-1]]
        yerror  = [np.sqrt(uproot.open(f'{inputdir}/inputs_{b}.root:{cat}').values()) for cat in args.cat_order[::-1]]
        content_up   = [uproot.open(f'{inputdir}/inputs_{b}.root:{cat}_{unce_type}Up').values() for cat in args.cat_order[::-1]]
        content_down = [uproot.open(f'{inputdir}/inputs_{b}.root:{cat}_{unce_type}Down').values() for cat in args.cat_order[::-1]]
        lab_suf = ''
        if norm_unce:
            lab_suf = '(norm)'
            for icat, cat in enumerate(args.cat_order[::-1]):
                content_up[icat] *= content[icat].sum() / content_up[icat].sum()
                content_down[icat] *= content[icat].sum() / content_down[icat].sum()
        f, ax = plt.subplots(figsize=(12, 12))
        hep.cms.label(data=True, llabel='Preliminary', year=args.year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % args.lumi, fontname='sans-serif')
        for icat, (cat, color) in enumerate(zip(args.cat_order[::-1], ['blue', 'red', 'green'])):
            hep.histplot(content[icat], yerr=yerror[icat], bins=edges, label=f'MC ({cat})', color=color)
        for icat, (cat, color) in enumerate(zip(args.cat_order[::-1], ['blue', 'red', 'green'])):
            hep.histplot(content_up[icat], bins=edges, label=f'MC ({cat}) {unce_type}Up {lab_suf}', color=color, linestyle='--')
        for icat, (cat, color) in enumerate(zip(args.cat_order[::-1], ['blue', 'red', 'green'])):
            hep.histplot(content_down[icat], bins=edges, label=f'MC ({cat}) {unce_type}Down {lab_suf}', color=color, linestyle=':')
        ax.set_xlim(xmin, xmax); ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlabel(args.xlabel, ha='right', x=1.0); ax.set_ylabel('Events / bin', ha='right', y=1.0)
        ax.legend(prop={'size': 18})
        
        if save_plots:
            plt.savefig(f'{workdir}/unce_comp_{unce_type}_{b}.png')
            plt.savefig(f'{workdir}/unce_comp_{unce_type}_{b}.pdf')
            plt.close()


def make_generic_mc_data_plots(
    edges,
    values_mc_list: list, yerr_mctot, 
    values_data, yerrlo_data, yerrhi_data,
    labels_mc, colors_mc, xlabel, ylabel, year, lumi,
    **kwargs
):
    r"""Make generic plots to show MC stacked and data distributions including a ratio subfigure.

    Arguments:
        edges: edges of the histogram
        values_mc_list: list of numpy arrays of MC bin contents for each MC components (colored stack areas)
        yerr_mctot: numpy array of MC total bin-wise uncertainties (length of hatched errorbar)
        values_data: numpy array of data bin content (black dot)
        yerrlo_data: numpy array of data lower edge uncertainty (errorbar on black dot)
        yerrhi_data: numpy array of data high edge uncertainty (errorbar on black dot)
        labels_mc: list of MC label strings (from bottom to top)
        colors_mc: list of MC color indices (from bottom to top)
    """
    def set_sns_color(*args):
        sns.palplot(sns.color_palette(*args))
        sns.set_palette(*args)

    set_sns_color(colors_mc)
    use_helvetica = kwargs.get('use_helvetica', False)
    if use_helvetica == True or (use_helvetica == 'auto' and any(['Helvetica' in font for font in mpl.font_manager.findSystemFonts()])):
        plt.style.use({"font.sans-serif": 'Helvetica'})

    f = plt.figure(figsize=(12, 12))
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05) 

    ## Upper histogram panel
    ax = f.add_subplot(gs[0])
    hep.cms.label(data=True, llabel='Preliminary', year=year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % lumi, fontname='sans-serif')
    xmin, xmax = min(edges), max(edges)
    ax.set_xlim(xmin, xmax); ax.set_xticklabels([]); 
    ax.set_ylabel(ylabel, ha='right', y=1.0)

    # draw total BGK uncetainty (hatched)
    hep.histplot(values_mc_list, bins=edges, label=labels_mc, histtype='fill', edgecolor='k', linewidth=1, stack=True) ## draw MC
    values_mctot = sum(values_mc_list)
    ax.fill_between(edges, (values_mctot-yerr_mctot).tolist()+[0], (values_mctot+yerr_mctot).tolist()+[0], label='MC total unce.', 
                    step='post', hatch='\\\\', edgecolor='dimgrey', facecolor='none', linewidth=0
                    )
    # draw data with errorbar
    hep.histplot(values_data, yerr=(yerrlo_data, yerrhi_data), 
                 bins=edges, label='Data', histtype='errorbar', color='k', markersize=15, elinewidth=1.5
                 )
    ax.set_ylim(0, 1.8*max(values_data))
    ax.legend()

    ## Ratio panel
    ax1 = f.add_subplot(gs[1]); ax1.set_xlim(xmin, xmax); ax1.set_ylim(0.001, 1.999)
    ax1.set_xlabel(xlabel, ha='right', x=1.0); ax1.set_ylabel('Data / MC', ha='center')
    ax1.plot([xmin, xmax], [1, 1], 'k'); ax1.plot([xmin, xmax], [0.5, 0.5], 'k:'); ax1.plot([xmin, xmax], [1.5, 1.5], 'k:')

    values_mctot_clip = np.maximum(values_mctot, 1e-20)
    ax1.fill_between(edges, ((values_mctot-yerr_mctot)/values_mctot_clip).tolist()+[0], ((values_mctot+yerr_mctot)/values_mctot_clip).tolist()+[0],
                    step='post', hatch='\\\\', edgecolor='dimgrey', facecolor='none', linewidth=0
                    ) ## draw bkg unce.
    hep.histplot(values_data/values_mctot_clip, yerr=(yerrlo_data/values_mctot_clip, yerrhi_data/values_mctot_clip), 
                 bins=edges, histtype='errorbar', color='k', markersize=15, elinewidth=1
                 )
    return f


############################################
## Below are utilities for plotting the SFs

def make_sfbdt_variation_plot(center, errl, errh, c_idx, sf, outputdir, args, plot_text, plot_name):
    r"""Summarize the SFs for different sfBDT coastline selections in one plot.

    Arguments:
        center, errl, errh: list of array of SFs with errorbars
        c_idx, index of the central SF in the list
        sf: SF type (bb or cc) that controls the color and range of the plot
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    if sf == args.cat_order[-1][-1]:
        ymin, ymax, facecolor = 0.5, 1.5, 'yellow'
    elif sf == args.cat_order[-2][-1]:
        ymin, ymax, facecolor = 0., 3.0, 'greenyellow'
    elif sf == args.cat_order[-3][-1]:
        ymin, ymax, facecolor = 0., 3.0, 'skyblue'

    def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r', edgecolor=None, alpha=0.5, label=None):
        errorboxes = []
        for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
            rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
            errorboxes.append(rect)
        pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)
        ax.add_collection(pc)
        artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror, fmt='None', ecolor=facecolor, label=label)
        return artists

    if args.use_helvetica:
        plt.style.use({"font.sans-serif": 'Helvetica'})

    # Plot SF points with errorbars
    f, ax = plt.subplots(figsize=(20, 10))
    hep.cms.label(data=True, llabel='Preliminary', year=args.year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % args.lumi, fontname='sans-serif')
    x_ticks = list(np.arange(len(center)))
    n_csl = int(round(np.sqrt(len(center))))
    x_ticklabels = ['' for _ in x_ticks]
    for i in range(n_csl):
        x_ticklabels[i * n_csl] = r'$c_{%d}c_{%d}$' % (i+1, 1)
        if i == n_csl - 1:
            x_ticklabels[i * n_csl + (n_csl - 1)] = r'$c_{%d}c_{%d}$' % (i+1, n_csl)
        if n_csl >= 3:
            x_ticklabels[i * n_csl + int((n_csl - 1)/2)] = r'$c_{%d}c_{j\cdots}$' % (i+1)
        if i > 0: # draw separating lines
            ax.plot([i*n_csl - 0.5, i*n_csl - 0.5], [ymin, ymax], color='red', linestyle='dotted')
    
    ax.plot([min(x_ticks), max(x_ticks)], [1, 1], color='grey', linestyle='dashed')
    ax.errorbar(x_ticks, center, yerr=[-np.array(errl),np.array(errh)], color='k', marker='s', markersize=8, linestyle='none', label=r'$SF(flv%s)\pm unce.$' % sf)
    ax.fill_between(x_ticks, np.array(center)+np.array(errl), np.array(center)+np.array(errh), edgecolor='darkblue', facecolor=facecolor, linewidth=0) ## draw bkg unce.
    
    # Plot central box
    if c_idx != 0 and c_idx != len(x_ticks) - 1:
        box_width_l, box_width_r = (x_ticks[c_idx]-x_ticks[c_idx - 1]) / 2, (x_ticks[c_idx + 1]-x_ticks[c_idx]) / 2
        margin = x_ticks[1] - x_ticks[0]
    else:
        box_width_l = box_width_r = margin = 0.1
    make_error_boxes(ax, xdata=[x_ticks[c_idx]], ydata=[center[c_idx]],
                    xerror=np.array([[box_width_l], [box_width_r]]),
                    yerror=np.array([[-errl[c_idx]], [errh[c_idx]]]), 
                    alpha=0.2, facecolor='red', label=r'$SF(flv%s)\pm unce.$ (central)' % sf
                    ) 
    ax.legend()
    ax.set_xlim(min(x_ticks)-margin, max(x_ticks)+margin); ax.set_ylim(ymin, ymax); ax.set_xlabel('sfBDT coastline selection (i: pass, j: fail)', ha='right', x=1.0)
    ax.set_xticks(x_ticks); ax.set_xticklabels(x_ticklabels); ax.tick_params(axis='both', which='minor', bottom=False, top=False)
    ax.text(0.10, 0.10, plot_text[0], fontweight='bold', transform=ax.transAxes)
    ax.text(0.30, 0.10, plot_text[1], transform=ax.transAxes)
    plt.savefig(os.path.join(outputdir, plot_name + '.png'))
    plt.savefig(os.path.join(outputdir, plot_name + '.pdf'))
    plt.close()


def make_fit_summary_plots(center, errl, errh, outputdir, args, plot_xticklabels, plot_ylabel, plot_legends, plot_text, plot_name):
    r"""Summarize the SFs in pT and WPs, for a specific fit scheme, in one plot.

    Arguments:
        center, errl, errh: 2D list of array of SFs with errorbars. dim1: pT bins, dim2: WPs
        sf: SF type (bb or cc) that controls the color and range of the plot
    """

    custom_cycler = (cycler(color=['darkblue', 'red', 'green', 'darkorange', 'cyan', 'magenta']) + \
        cycler(marker=['s', 'P', 'o', 'd', 'X', '*']))

    x_ticks = np.arange(len(plot_xticklabels))  # the label locations
    width = 0.15  # the width of the bars

    if args.use_helvetica:
        plt.style.use({"font.sans-serif": 'Helvetica'})

    f, ax = plt.subplots(figsize=(10, 10))
    hep.cms.label(data=True, llabel='Preliminary', year=args.year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)' % args.lumi, fontname='sans-serif')
    ax.set_prop_cycle(custom_cycler)
    for yl in np.arange(0, 2.4, 0.2):
        ax.plot([-0.5, len(plot_xticklabels)+1], [yl, yl], ':', color='lightgrey')
    ax.plot([-0.5, len(plot_xticklabels)+1], [1., 1.], ':', color='grey')

    for i, (c, el, eh) in enumerate(zip(center, errl, errh)):
        ax.errorbar(x_ticks + (i/2.-1)*width, c, yerr=[-np.array(el), np.array(eh)], markersize=8, linestyle='none', label=plot_legends[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(r'$p_{T}$(j) [GeV]', ha='right', x=1.0); ax.set_ylabel(plot_ylabel, ha='right', y=1.0)
    ax.set_xlim(-0.5, len(plot_xticklabels)-0.5); ax.set_ylim(0.5, 1.5)
    ax.set_xticks(x_ticks); ax.set_xticklabels(plot_xticklabels); ax.tick_params(axis='both', which='minor', bottom=False, top=False)
    ax.text(0.05, 0.92, plot_text, transform=ax.transAxes, fontweight='bold') 
    ax.legend(loc='lower left')

    plt.savefig(os.path.join(outputdir, plot_name + '.png'))
    plt.savefig(os.path.join(outputdir, plot_name + '.pdf'))
    plt.close()
