import boost_histogram as bh

def bh_to_uproot(histogram: bh.Histogram):
    r"""Return a boost histogram object that uproot5 can write as TH1.
    """

    return histogram


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
