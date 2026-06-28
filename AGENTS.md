# Agent Notes

- Before running project commands, load the default environment:
  ```bash
  source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt/setup.sh
  ```

- For launcher jobs or batch plotting, set writable cache locations:
  ```bash
  MPLCONFIGDIR=/tmp/mpl-topwsf
  PYTHONPYCACHEPREFIX=/tmp/pycache-topwsf
  ```

- For `step(..., where="post")` plots, do not draw with `edges[:-1]` and `values`.
  The last bin will visually disappear. Use:
  ```python
  ax.step(edges, np.r_[values, values[-1]], where="post")
  ```

- `WebMaker.add_figure()` inserts raw `<img>` tags. Add blank `add_text()` lines
  after text and around figure groups so the webpage layout stays readable.

- Standard Data/MC and MC-only distributions are usually drawn with
  `plt.subplots(figsize=(10, 10))`. Show them as square figures in webpages too,
  for example with matching `width` and `height` in `WebMaker.add_figure()`.

- When making a git commit, use a concise subject line and include bullet
  points in the commit body describing the concrete changes.
