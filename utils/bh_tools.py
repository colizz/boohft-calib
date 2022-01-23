from uproot3_methods.classes.TH1 import _histtype, Methods
import boost_histogram as bh
import numpy as np

def bh_to_uproot3(histogram: bh.Histogram):
    r"""
    Write a boost histogram to ROOT file using uproot3 as backend, extended from
      https://github.com/scikit-hep/uproot3-methods/blob/master/uproot3_methods/classes/TH1.py
    Note that uproot(4) will soon officially support boost histogram writing but here we extend the method
      in uproot3_methods as a fast and temporary implementation
    """

    values = histogram.values()
    allvalues = histogram.values(flow=True) # flows included, with 2 more bins
    allvariances = histogram.variances(flow=True)
    edges = histogram.axes[0].edges

    class TH1(Methods, list):
        pass

    class TAxis(object):
        def __init__(self, edges):
            self._fNbins = len(edges) - 1
            self._fXmin = edges[0]
            self._fXmax = edges[-1]
            if np.array_equal(edges, np.linspace(self._fXmin, self._fXmax, len(edges), dtype=edges.dtype)):
                self._fXbins = np.array([], dtype=">f8")
            else:
                self._fXbins = edges.astype(">f8")

    out = TH1.__new__(TH1)
    out._fXaxis = TAxis(edges)

    centers = (edges[:-1] + edges[1:]) / 2.0
    out._fEntries = out._fTsumw = out._fTsumw2 = values.sum()
    out._fTsumwx = (values * centers).sum()
    out._fTsumwx2 = (values * centers**2).sum()

    out._fTitle = histogram.label

    out._classname, values = _histtype(values)

    out.extend(allvalues)
    out._fSumw2 = allvariances

    return out


def fix_bh(h_in: bh.Histogram, eps=1e-3) -> bh.Histogram:
    r"""Fix boost histogram template for writing
        1) Merge flows into the first / last bins to follow the ROOT convention
        2) Fix the negative bins and large variance
    """

    h_out = h_in.copy()
    # absorb over/underflow into the first and last bin
    h_out[0] += h_out[bh.underflow]
    h_out[-1] += h_out[bh.overflow]
    h_out[bh.underflow] = (0., 0.)
    h_out[bh.overflow] = (0., 0.)
    # fix the bin contents inplace
    for i in range(len(h_out.view(flow=True))):
        if h_out.values(flow=True)[i] < eps:
            h_out.values(flow=True)[i] = eps
            h_out.variances(flow=True)[i] = eps
        elif h_out.variances(flow=True)[i] > h_out.values(flow=True)[i] ** 2:
            h_out.variances(flow=True)[i] = h_out.values(flow=True)[i] ** 2

    return h_out


def scale_bh(h_in: bh.Histogram, weight) -> bh.Histogram:
    r"""Scale the boost histogram with a given weight or a binwise weight template (flows included)
    """

    h_out = h_in.copy()
    if isinstance(weight, (float, int)):
        weight = [weight for b in h_out.view(flow=True)]
    for i in range(len(h_out.view(flow=True))):
        h_out.values(flow=True)[i] *= weight[i]
        h_out.variances(flow=True)[i] *= (weight[i] ** 2)

    return h_out
