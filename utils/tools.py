import awkward as ak
import numpy as np

from logger import _logger

def lookup_pt_based_weight(weight_map, pt_reweight_edges, jet_idx, jet_pt, jet_var, jet_var_maxlimit=None):
    r"""Obtain the pT-based weight using the weight map. jet_idx: 'fj1' or 'fj2'."""

    assert jet_var_maxlimit is not None, "Need to specify a jet_var_maxlimit."
    # flatten the 2d weight factor map
    weight_map_flatten = ak.flatten(ak.Array([weight_map[c]['h_w'] for c in weight_map if c.startswith(jet_idx)]))
    var_edges_list = [([0] + weight_map[c]['edges']) for c in weight_map if c.startswith(jet_idx)]
    flatten_edges = [edge + ivar * jet_var_maxlimit for ivar, var_edges in enumerate(var_edges_list) for edge in var_edges]

    # get the 1d index on the flattened factor map
    pt_pos = np.maximum(np.searchsorted(pt_reweight_edges, ak.to_numpy(jet_pt), side='right') - 1, 0) # note: first bin i.e. pT in [200, 250) has index 0

    reweight_flatten_var = pt_pos * jet_var_maxlimit + np.minimum(jet_var, jet_var_maxlimit)
    reweight_flatten_pos = np.searchsorted(flatten_edges, ak.to_numpy(reweight_flatten_var), side='right') - 1

    # index the reweight factors
    weight = weight_map_flatten[reweight_flatten_pos]
    # _logger.debug(f"pT based weights: {jet_idx=}, {jet_var[:20]=}, {jet_pt[:20]=}, {weight[:20]=}")

    return weight


def parse_tagger_expr(tagger_name_replace_map, expr):
    r"""Parse the tagger expression and replace NanoAOD varaibles into HRT tuples format."""

    import ast
    root = ast.parse(expr)
    replace_dict = {}
    for node in ast.walk(root):
        if isinstance(node, ast.Name) and not node.id.startswith('_'):
            if node.id not in tagger_name_replace_map:
                _logger.exception('Tagger variable \'{node}\' is not supported in our method')
                raise
            replace_dict[node.id] = tagger_name_replace_map[node.id]

    for var, var_replace in replace_dict.items():
        expr = expr.replace(var, var_replace)
    return expr
