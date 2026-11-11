% Regenerates tests/fixtures/mst_matlab.mat: what MST_tree.m builds, for
% tests/test_matlab_mst.py to compare against. Needs the TREES toolbox on the
% path. Each parent array holds, per point, the input index of its parent
% (seed-tree nodes as negative node indices, a tree's start point as
% -tree index for multi-start runs, 0 if the point was never attached).

rng (1);
out = struct;
for s = 1 : 3
    P = 100 * rand (200, 3);
    out.(sprintf ('P_%d', s)) = P;

    % one start point
    cases = {'ctrl', 0.4, 1e4, ''; 'b', 0.4, 1e4, '-b'; 'm02', 0.2, 60, ''; 'm07', 0.7, 60, ''};
    for c = 1 : size (cases, 1)
        [tr, ix] = MST_tree (1, P (:, 1), P (:, 2), P (:, 3), cases {c, 2}, 50, cases {c, 3}, [], cases {c, 4});
        idp = idpar_tree (tr);
        n2p = zeros (size (tr.X));  n2p (1) = 1;
        hit = find (ix (:, 2) > 0);
        n2p (ix (hit, 2)) = hit + 1;        % ix covers the 199 points left after the start
        par = zeros (200, 1);  par (1) = -1;
        for h = hit'
            par (h + 1) = n2p (idp (ix (h, 2)));
        end
        out.(sprintf ('par_%s_%d', cases {c, 1}, s)) = par;
    end

    % growth onto an existing tree
    seed = MST_tree (1, P (1 : 30, 1), P (1 : 30, 2), P (1 : 30, 3), 0.4, 50, 1e4, [], '');
    out.(sprintf ('seed_%d', s)) = [seed.X seed.Y seed.Z idpar_tree(seed, '-z')];
    Q = P (31 : end, :);
    ns = numel (seed.X);
    cases = {'sd', 1e4, ''; 'sb', 1e4, '-b'; 'sm', 90, ''};
    for c = 1 : size (cases, 1)
        [tr, ix] = MST_tree ({seed}, Q (:, 1), Q (:, 2), Q (:, 3), 0.4, 50, cases {c, 2}, [], cases {c, 3});
        if iscell (tr), tr = tr {1}; end
        idp = idpar_tree (tr);
        n2p = zeros (size (tr.X));  n2p (1 : ns) = -(1 : ns);
        hit = find (ix (:, 2) > 0);
        n2p (ix (hit, 2)) = hit;
        par = zeros (size (Q, 1), 1);
        for h = hit'
            par (h) = n2p (idp (ix (h, 2)));
        end
        out.(sprintf ('par_%s_%d', cases {c, 1}, s)) = par;
    end

    % several start points
    for c = {{'mt', ''}, {'mb', '-b'}}
        [trs, ix] = MST_tree ([1 2 3], P (:, 1), P (:, 2), P (:, 3), 0.4, 50, 1e4, [], c {1} {2});
        par = zeros (197, 1);
        for h = find (ix (:, 2) > 0)'
            t = ix (h, 1);  q = idpar_tree (trs {t}, '-z');  q = q (ix (h, 2));
            if q == 1
                par (h) = -t;
            else
                par (h) = find (ix (:, 1) == t & ix (:, 2) == q, 1);
            end
        end
        out.(sprintf ('par_%s_%d', c {1} {1}, s)) = par;
    end
end
save ('mst_matlab.mat', '-struct', 'out', '-v7');
