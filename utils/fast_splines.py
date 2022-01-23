#########################################################
# This file is under the Apache License, Version 2.0
# 
# original code: 
#   https://github.com/dbstein/fast_splines/blob/master/fast_splines/fast_splines.py
# 
#########################################################

import numpy as np
import scipy as sp
import scipy.interpolate
import numba

class interp2d(object):
    def __init__(self, xv, yv, z, k=3):
        """
        xv are the x-data nodes, in strictly increasing order
        yv are the y-data nodes, in strictly increasing order
            both of these must be equispaced!
        (though x, y spacing need not be the same)
        z is the data
        k is the order of the splines (int)
            order k splines give interp accuracy of order k+1
            only 1, 3, 5, supported
        """
        if k not in [1, 3, 5]:
            raise Exception('k must be 1, 3, or 5')
        self.xv = xv
        self.yv = yv
        self.z = z
        self.k = k
        self._dtype = yv.dtype
        terp = sp.interpolate.RectBivariateSpline(xv, yv, z, kx=k, ky=k)
        self._tx, self._ty, self._c = terp.tck
        self._nx = self._tx.shape[0]
        self._ny = self._ty.shape[0]
        self._hx = self.xv[1] - self.xv[0]
        self._hy = self.yv[1] - self.yv[0]
        self._nnx = self.xv.shape[0]-1
        self._nny = self.yv.shape[0]-1
        self._cr = self._c.reshape(self._nnx+1, self._nny+1)
    def __call__(self, op_x, op_y, out=None):
        """
        out_points are the 1d array of x values to interp to
        out is a place to store the result
        """
        m = int(np.prod(op_x.shape))
        copy_made = False
        if out is None:
            _out = np.empty(m, dtype=self._dtype)
        else:
            # hopefully this doesn't make a copy
            _out = out.ravel()
            if _out.base is None:
                copy_made = True
        _op_x = op_x.ravel()
        _op_y = op_y.ravel()
        splev2(self._tx, self._nx, self._ty, self._ny, self._cr, self.k, \
            _op_x, _op_y, m, _out, self._hx, self._hy, self._nnx, self._nny)
        _out = _out.reshape(op_x.shape)
        if copy_made:
            # if we had to make a copy, update the provided output array
            out[:] = _out
        return _out

@numba.njit(parallel=True)
def splev2(tx, nx, ty, ny, c, k, x, y, m, z, dx, dy, nnx, nny):
    # fill in the h values for x
    k1 = k+1
    hbx = np.empty((m, 6))
    hhbx = np.empty((m, 5))
    lxs = np.empty(m, dtype=np.int64)
    splev_short(tx, nx, k, x, m, dx, nnx, hbx, hhbx, lxs)
    hby = np.empty((m, 6))
    hhby = np.empty((m, 5))
    lys = np.empty(m, dtype=np.int64)
    splev_short(ty, ny, k, y, m, dy, nny, hby, hhby, lys)
    for i in numba.prange(m):
        sp = 0.0
        llx = lxs[i] - k1
        for j in range(k1):
            llx += 1
            lly = lys[i] - k1
            for k in range(k1):
                lly += 1
                sp += c[llx,lly] * hbx[i,j] * hby[i,k]
        z[i] = sp

@numba.njit(parallel=True)
def splev_short(t, n, k, x, m, dx, nn, hb, hhb, lxs):
    # fetch tb and te, the boundaries of the approximation interval
    k1 = k+1
    nk1 = n-k1
    tb = t[k1-1]
    te = t[nk1+1-1]
    l = k1
    l1 = l+1
    adj = int(k/2) + 1
    # main loop for the different points
    for i in numba.prange(m):
        h = hb[i]
        hh = hhb[i]
        # fetch a new x-value arg
        arg = x[i]
        arg = max(tb, arg)
        arg = min(te, arg)
        # search for knot interval t[l] <= arg <= t[l+1]
        l = int(arg/dx) + adj
        l = max(l, k)
        l = min(l, nn)
        lxs[i] = l
        # evaluate the non-zero b-splines at arg.
        h[0] = 1.0
        for j in range(k):
            for ll in range(j+1):
                hh[ll] = h[ll]
            h[0] = 0.0
            for ll in range(j+1):
                li = l + ll + 1
                lj = li - j - 1
                f = hh[ll]/(t[li]-t[lj])
                h[ll] += f*(t[li]-arg)
                h[ll+1] = f*(arg-t[lj])