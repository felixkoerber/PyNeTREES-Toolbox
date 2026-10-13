function run_boundary_validation(outdir)
%RUN_BOUNDARY_VALIDATION  Capture MATLAB's own numbers for the alpha shape
%   and everything downstream of it.
%
%   run_boundary_validation(OUTDIR) reads OUTDIR/cells.mtr -- written from
%   Python with `pt.save_mtr([sample_tree, hsn_tree, hss_tree], ...)` -- and
%   prints, then saves, the reference values pinned in
%   tests/test_matlab_boundary.py.
%
%   Why this exists: `boundary()` is the one dependency Octave does not
%   implement, so until MATLAB was installed here, `boundary_tree`,
%   `r_mc_tree`, `theta_mc_tree`, `growth_tree` and `convexity_tree` had no
%   differential check at all. `r_mc_tree`, `theta_mc_tree` and
%   `growth_tree` additionally reach `boundary()`'s output through the File
%   Exchange routine `inpolyhedron`, which is why that file sits beside this
%   one -- without it those three cannot be run on the MATLAB side at all.
%
%   Put this directory and the TREES toolbox on the path, then:
%
%       matlab -batch "run_boundary_validation('<scratch dir>')"

here = fileparts(mfilename('fullpath'));
addpath(here);                                   % inpolyhedron
if isempty(which('T_tree'))
    error(['run_boundary_validation: the TREES toolbox is not on the ' ...
           'path. addpath(genpath(<treestoolbox>)) first.']);
end

L      = load(fullfile(outdir, 'cells.mtr'), '-mat');
names  = {'sample', 'hsn', 'hss'};
alphas = [0 0.25 0.5 0.75 1.0];
res    = struct();

%% 1. the shape ---------------------------------------------------------
% `boundary.m` is a readable m-file: it interpolates a rank within
% alphaSpectrum, anchored at criticalAlpha(shp,'one-region'). Both critical
% alphas are dumped because the port originally anchored on the wrong one,
% and they coincide on two of these three cells -- which is exactly why
% that went unnoticed.
fprintf('\n== boundary() ==\n');
for ci = 1:numel(names)
    tree = L.tree{ci};
    V    = [tree.X(:) tree.Y(:) tree.Z(:)];
    shp  = alphaShape(V, Inf);
    fprintf('%-8s one-region %.15g   all-points %.15g\n', ...
            names{ci}, criticalAlpha(shp, 'one-region'), ...
            criticalAlpha(shp, 'all-points'));
    for ai = 1:numel(alphas)
        [bd3, v3] = boundary(V(:,1), V(:,2), V(:,3), alphas(ai));
        [~,   a2] = boundary(V(:,1), V(:,2),         alphas(ai));
        tag = sprintf('%s_a%03d', names{ci}, round(alphas(ai) * 100));
        fprintf('  shrink %.2f: volume %.15g  facets %d  area2d %.15g\n', ...
                alphas(ai), v3, size(bd3, 1), a2);
        res.([tag '_volume']) = v3;
        res.([tag '_facets']) = size(bd3, 1);
        res.([tag '_area2d']) = a2;
    end
end

%% 2. the statistics that reach it through inpolyhedron -----------------
fprintf('\n== downstream of inpolyhedron ==\n');
for ci = 1:2                                     % hss is slow and adds nothing
    tree = L.tree{ci};
    name = names{ci};

    rng(12345);
    theta = theta_mc_tree(tree, 200000, 0.5, '');
    fprintf('%-8s theta_mc (NN 2e5, alpha .5) %.15g\n', name, theta);
    res.([name '_theta']) = theta;

    % n_mc is deliberately large: at 20 iterations the two samplers differ
    % by 1-2%, which is noise, and only converging shows that it is.
    rng(12345);
    o  = r_mc_tree(tree, 'alpha', 0.5, 'n_mc', 400);
    % NOTE '-nv' is documented as "no volume correction" but the code reads
    % `if pars.nv % volume correction`, so this is the *corrected* branch.
    rng(12345);
    o2 = r_mc_tree(tree, 'alpha', 0.5, 'n_mc', 400, 'nv', true);
    fprintf(['%-8s r_mc  r0 %.15g   R %.15g (uncorrected)   ' ...
             'R %.15g (corrected)\n'], name, o.r0, o.R, o2.R);
    res.([name '_r0'])    = o.r0;
    res.([name '_R'])     = o.R;
    res.([name '_R_corr'])= o2.R;
end

%% 3. convexity_tree ----------------------------------------------------
% No inpolyhedron, but the other consumer of boundary(), and the reason the
% flag form is spelled out here is bug 1.14: '-dim2' does not reach the 2D
% branch, because parseArgs never clears dim3's default of true.
fprintf('\n== convexity_tree ==\n');
for ci = 1:2
    tree = L.tree{ci};
    name = names{ci};
    c3 = convexity_tree(tree, '-dim3');
    cf = convexity_tree(tree, '-dim2');                       % ignored
    c2 = convexity_tree(tree, 'dim2', true, 'dim3', false);   % actually 2D
    fprintf('%-8s dim3 %.15g | ''-dim2'' %.15g | forced 2D %.15g | 1/nT %.15g\n', ...
            name, c3, cf, c2, 1 / sum(T_tree(tree)));
    res.([name '_c3'])       = c3;
    res.([name '_c_dim2flag'])= cf;
    res.([name '_c2'])       = c2;
    res.([name '_nT'])       = sum(T_tree(tree));
end

save(fullfile(outdir, 'matlab_boundary_reference.mat'), '-struct', 'res');
fprintf('\nwrote matlab_boundary_reference.mat\n');
end
