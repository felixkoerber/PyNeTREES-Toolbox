# MATLAB-side helpers for differential testing

Not part of the Python package. These exist so that a claim like "the port
matches MATLAB" can be *checked* rather than asserted, and they are kept here
so the next person does not have to rediscover which File Exchange downloads
the reference implementation needs.

## `inpolyhedron.m`

A File Exchange dependency of the MATLAB TREES toolbox — `r_mc_tree`,
`theta_mc_tree` and `growth_tree`'s `mc3d` branch all call it, and it does
not ship with TREES. Without it those three cannot be run at all on the
MATLAB side, so neither can they be compared against.

Put this on the MATLAB path (`addpath` this directory) before running any
harness that touches them.

## Running a comparison

MATLAB is driven headless:

```bash
"C:/Program Files/MATLAB/R2026a/bin/matlab.exe" -batch "run_mst('<scratch dir>')"
```

The pattern that works, from the `MST_tree` comparison in 0.2.0:

1. Generate fixed inputs in Python, write them to `.mat` with `scipy.io.savemat`.
2. Run the MATLAB reference on exactly those inputs, dump its output to `.mat`.
3. Compare in Python — and compare **structure at matched coordinates**, not
   node indices. Both sides visit a point cloud in their own order, so node 7
   need not be the same point in both; matching by rounded coordinate makes
   the parent relation comparable regardless. That one detail is what made
   the `MST_tree` comparison tractable.

The reference toolbox this port is checked against lives outside the repo;
see `PORT_STATUS.md` Design Decision #69 for which release and where.
