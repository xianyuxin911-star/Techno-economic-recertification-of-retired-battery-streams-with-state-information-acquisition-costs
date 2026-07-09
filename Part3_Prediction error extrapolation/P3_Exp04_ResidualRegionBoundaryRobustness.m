%% P3_Exp04_ResidualRegionBoundaryRobustness.m
% Purpose:
% This script evaluates the robustness of the residual-distribution
% extrapolation model to residual-region boundary perturbations and
% generates Supplementary Fig. 12.
%
% The analysis uses all-inner residual samples from the nested-resampling
% experiment. The widths of the main and right-tail residual regions are
% narrowed or widened by 20%, and the resulting trends and goodness-of-fit
% values of P_M(k), P_R(k), and sigma_R(k) are compared.


clc; clear; close all;

%% 1) Configuration
cfg = struct();

script_dir = fileparts(mfilename('fullpath'));

if isempty(script_dir)
    script_dir = pwd;
end

cfg.part3_dir = script_dir;

cfg.output_dir = fullfile(cfg.part3_dir, 'Output');
cfg.results_dir = fullfile(cfg.output_dir, 'Results');

cfg.figure_root = fullfile(cfg.part3_dir, 'Figures');
cfg.supp_fig_dir = fullfile(cfg.figure_root, 'Supplementary');

cfg.input_mat = fullfile(cfg.results_dir, 'P3_Exp01_NestedResampling_Final.mat');

cfg.dist_dir = fullfile(cfg.supp_fig_dir,  'SupFig12_residual_region_distributions_k05_to_k25');
cfg.fit_summary_csv = fullfile(cfg.results_dir, 'P3_Exp04_ResidualRegionBoundaryRobustness_FitSummary.csv');
cfg.feature_curves_csv = fullfile(cfg.results_dir, 'P3_Exp04_ResidualRegionBoundaryRobustness_FeatureCurves.csv');

cfg.fit_curves_csv = fullfile(cfg.results_dir, 'P3_Exp04_ResidualRegionBoundaryRobustness_FitCurves.csv');
cfg.trend_check_csv = fullfile(cfg.results_dir,  'P3_Exp04_ResidualRegionBoundaryRobustness_TrendCheck.csv');
cfg.all_inner_counts_csv = fullfile(cfg.results_dir,   'P3_Exp04_ResidualRegionBoundaryRobustness_AllInnerCounts.csv');
cfg.model_mat = fullfile(cfg.results_dir, 'P3_Exp04_ResidualRegionBoundaryRobustness_Model.mat');

cfg.fig_r2_summary = fullfile(cfg.supp_fig_dir,  'SupFig12a_ResidualRegionBoundary_R2_summary.fig');
cfg.png_r2_summary = fullfile(cfg.supp_fig_dir,  'SupFig12a_ResidualRegionBoundary_R2_summary.png');

cfg.fig_PM = fullfile(cfg.supp_fig_dir,  'SupFig12b_ResidualRegionBoundary_PM_sensitivity.fig');
cfg.png_PM = fullfile(cfg.supp_fig_dir,  'SupFig12b_ResidualRegionBoundary_PM_sensitivity.png');

cfg.fig_PR = fullfile(cfg.supp_fig_dir,  'SupFig12c_ResidualRegionBoundary_PR_sensitivity.fig');
cfg.png_PR = fullfile(cfg.supp_fig_dir,  'SupFig12c_ResidualRegionBoundary_PR_sensitivity.png');

cfg.fig_sigR = fullfile(cfg.supp_fig_dir,  'SupFig12d_ResidualRegionBoundary_SigmaR_sensitivity.fig');
cfg.png_sigR = fullfile(cfg.supp_fig_dir,  'SupFig12d_ResidualRegionBoundary_SigmaR_sensitivity.png');

cfg.png_resolution = 600;

folder_list = { ...
    cfg.output_dir, ...
    cfg.results_dir, ...
    cfg.figure_root, ...
    cfg.supp_fig_dir, ...
    cfg.dist_dir};

for ii = 1:numel(folder_list)
    if ~exist(folder_list{ii}, 'dir')
        mkdir(folder_list{ii});
    end
end

fprintf('================ P3 Exp04: residual-region boundary robustness ================\n');
fprintf('Input result file: %s\n', cfg.input_mat);
fprintf('Output folder:     %s\n', cfg.results_dir);
fprintf('Figure folder:     %s\n', cfg.supp_fig_dir);

%% 2) Basic settings
if ~exist(cfg.input_mat, 'file')
    error(['Input data file not found: %s\n' ...
        'Please run P3_Exp01_NestedResampling_ErrorEvaluation.m first to generate the nested-resampling result.'], ...
        cfg.input_mat);
end

% Empirical k range.
k_emp_min = 5;
k_emp_max = 21;
k_anchor = 21;

% Extrapolated k range.
k_extrap_max = 50;
k_vals = (k_emp_min:k_extrap_max)';

% Representative k values for residual-distribution visualization.
plot_k_list = [5, 10, 15, 20, 25];

% Number of Monte Carlo samples used for extrapolated distributions beyond k = 21.
N_sim_plot = 100000;
rng(42);

% Endpoint for bridging empirical and parametric distributions.
k_bridge_end = 26;

% Baseline residual-region definitions.
base_main   = [-0.012,  0.015];
base_right  = [ 0.028,  0.052];
base_valley = [ 0.015,  0.028];
base_left   = [-0.060, -0.020];

% KDE settings.
kde_bw = 0.0055;
hist_edges = linspace(-0.06, 0.06, 70);

do_plot_distribution = true;

%% 3) Define residual-region boundary perturbation scenarios
ScenarioName = [
    "Baseline"
    "Narrow_all_20"
    "Wide_all_20"
    "Narrow_main_20"
    "Wide_main_20"
    "Narrow_right_20"
    "Wide_right_20"
    ];

main_mult = [
    1.0
    0.8
    1.2
    0.8
    1.2
    1.0
    1.0
    ];

right_mult = [
    1.0
    0.8
    1.2
    1.0
    1.0
    0.8
    1.2
    ];

nScenario = numel(ScenarioName);

%% 4) Load all-inner residuals for each k
S = load(cfg.input_mat);

required_vars = { ...
    'train_size_list', ...
    'all_inner_errs'};

for i = 1:numel(required_vars)
    if ~isfield(S, required_vars{i})
        error('Required variable is missing from the data file: %s', required_vars{i});
    end
end

train_size_list = S.train_size_list(:);
all_inner_errs = S.all_inner_errs;

k_emp = (k_emp_min:k_emp_max)';
num_k_emp = numel(k_emp);

err_cell = cell(num_k_emp, 1);

num_outer_used_list = zeros(num_k_emp, 1);
num_inner_used_list = zeros(num_k_emp, 1);
num_residual_used_list = zeros(num_k_emp, 1);

for i = 1:num_k_emp

    k_now = k_emp(i);
    idx_k = find(train_size_list == k_now, 1);

    if isempty(idx_k)
        error('k = %d was not found in train_size_list.', k_now);
    end

    err_all = [];
    num_outer_used = 0;
    num_inner_used = 0;

    for outer_id = 1:size(all_inner_errs, 2)

        E = all_inner_errs{idx_k, outer_id};

        if isempty(E)
            continue;
        end

        E = double(E);

        if isempty(E) || all(~isfinite(E(:)))
            continue;
        end

        % Each column corresponds to one inner repetition.
        % Each row corresponds to one fixed test pack in the current outer split.
        num_outer_used = num_outer_used + 1;
        num_inner_used = num_inner_used + size(E, 2);

        E = E(isfinite(E));
        err_all = [err_all; E(:)]; %#ok<AGROW>
    end

    if isempty(err_all)
        error('No valid all-inner residual samples were found for k = %d.', k_now);
    end

    err_cell{i} = double(err_all(:));

    num_outer_used_list(i) = num_outer_used;
    num_inner_used_list(i) = num_inner_used;
    num_residual_used_list(i) = numel(err_all);

    fprintf('k = %d | outer runs = %d | inner runs = %d | all-inner residual samples = %d\n', ...
        k_now, ...
        num_outer_used_list(i), ...
        num_inner_used_list(i), ...
        num_residual_used_list(i));
end

AllInnerCountTable = table( ...
    k_emp, ...
    num_outer_used_list, ...
    num_inner_used_list, ...
    num_residual_used_list, ...
    'VariableNames', { ...
    'k', ...
    'N_outer_runs', ...
    'N_inner_runs_all_inner', ...
    'N_residuals_all_inner'});

writetable(AllInnerCountTable, cfg.all_inner_counts_csv);

fprintf('All-inner residual count table saved: %s\n', cfg.all_inner_counts_csv);

%% 5) Compute all-inner empirical RMSE and fit RMSE scaling law
rmse_emp = zeros(num_k_emp, 1);

for i = 1:num_k_emp
    e = err_cell{i};
    rmse_emp(i) = sqrt(mean(e.^2));
end

x_rmse = k_emp(:);
y_rmse = rmse_emp(:);

valid_rmse = isfinite(x_rmse) & isfinite(y_rmse) & (y_rmse > 0);
x_rmse = x_rmse(valid_rmse);
y_rmse = y_rmse(valid_rmse);

ft_RMSE = fittype('a * x^(-b) + c', ...
    'independent', 'x', ...
    'dependent', 'y');

opts_RMSE = fitoptions('Method', 'NonlinearLeastSquares');
opts_RMSE.Display = 'Off';
opts_RMSE.StartPoint = [0.1, 0.5, min(y_rmse)];
opts_RMSE.Lower = [0, 0, 0];

[fit_RMSE, gof_RMSE] = fit(x_rmse, y_rmse, ft_RMSE, opts_RMSE);

a_RMSE = fit_RMSE.a;
b_RMSE = fit_RMSE.b;
c_RMSE = fit_RMSE.c;
R2_RMSE = gof_RMSE.rsquare;

fprintf('\n================ All-inner RMSE scaling-law fitting ================\n');
fprintf('RMSE(k) = %.6f * k^(-%.6f) + %.6f\n', ...
    a_RMSE, b_RMSE, c_RMSE);
fprintf('RMSE scaling R2 = %.4f\n', R2_RMSE);
fprintf('====================================================================\n');

%% 6) Initialize result tables and storage structure
FitSummary_All = table();
FeatureCurve_All = table();
FitCurve_All = table();

Store = struct();

%% 7) Extract region features and refit the model for each scenario
for ss = 1:nScenario

    curr_name = ScenarioName(ss);

    % Generate residual-region boundaries for the current scenario.
    c_main = mean(base_main);
    hw_main = (base_main(2) - base_main(1)) / 2;

    region_main = [
        c_main - main_mult(ss) * hw_main, ...
        c_main + main_mult(ss) * hw_main
        ];

    c_right = mean(base_right);
    hw_right = (base_right(2) - base_right(1)) / 2;

    region_right = [
        c_right - right_mult(ss) * hw_right, ...
        c_right + right_mult(ss) * hw_right
        ];

    region_valley = base_valley;
    region_left = base_left;

    fprintf('\n======================================================\n');
    fprintf('Current scenario: %s\n', curr_name);
    fprintf('main  region = [%.5f, %.5f]\n', region_main(1), region_main(2));
    fprintf('right region = [%.5f, %.5f]\n', region_right(1), region_right(2));
    fprintf('======================================================\n');

    % Extract P_M, P_R, mu_M, mu_R, sig_M, and sig_R.
    P_M = zeros(num_k_emp, 1);
    P_R = zeros(num_k_emp, 1);
    P_V = zeros(num_k_emp, 1);
    P_L = zeros(num_k_emp, 1);

    mu_M = zeros(num_k_emp, 1);
    mu_R = zeros(num_k_emp, 1);
    sig_M = zeros(num_k_emp, 1);
    sig_R = zeros(num_k_emp, 1);
    rmseK = zeros(num_k_emp, 1);

    for ii = 1:num_k_emp

        e = err_cell{ii};

        if isempty(e)
            P_M(ii) = NaN;
            P_R(ii) = NaN;
            P_V(ii) = NaN;
            P_L(ii) = NaN;
            mu_M(ii) = NaN;
            mu_R(ii) = NaN;
            sig_M(ii) = NaN;
            sig_R(ii) = NaN;
            rmseK(ii) = NaN;
            continue;
        end

        rmseK(ii) = sqrt(mean(e.^2));

        idx_main   = (e >= region_main(1))   & (e < region_main(2));
        idx_right  = (e >= region_right(1))  & (e < region_right(2));
        idx_valley = (e >= region_valley(1)) & (e < region_valley(2));
        idx_left   = (e >= region_left(1))   & (e < region_left(2));

        P_M(ii) = mean(idx_main);
        P_R(ii) = mean(idx_right);
        P_V(ii) = mean(idx_valley);
        P_L(ii) = mean(idx_left);

        % Main-mode statistics.
        eM = e(idx_main);

        if isempty(eM)
            eM = e;
        end

        mu_M(ii) = median(eM);

        sM = iqr(eM) / 1.349;

        if numel(eM) < 4 || sM == 0 || isnan(sM)
            sM = std(eM);
        end

        sig_M(ii) = max(sM, 1e-4);

        % Right-tail statistics.
        eR = e(idx_right);

        if ~isempty(eR)

            mu_R(ii) = mean(eR);

            sR = iqr(eR) / 1.349;

            if numel(eR) < 4 || sR == 0 || isnan(sR)
                sR = std(eR);
            end

            sig_R(ii) = max(sR, 1e-4);

        else

            mu_R(ii) = mean(region_right);
            sig_R(ii) = 0.005;
        end
    end

    % Apply mild smoothing consistent with Main03.
    P_R   = min(max(smoothdata(P_R, 'movmean', 3), 0), 0.40);
    P_M   = min(max(smoothdata(P_M, 'movmean', 3), 0), 1.00);
    mu_M  = smoothdata(mu_M, 'movmean', 3);
    mu_R  = smoothdata(mu_R, 'movmean', 3);
    sig_M = max(smoothdata(sig_M, 'movmean', 3), 1e-4);
    sig_R = max(smoothdata(sig_R, 'movmean', 3), 1e-4);

    % Huber loss.
    huber_core = @(r, d) ...
        (abs(r) <= d) .* (0.5 * r.^2) + ...
        (abs(r) >  d) .* (d .* (abs(r) - 0.5 * d));

    huber_loss = @(y, yhat) ...
        sum(huber_core(y - yhat, 1.5 * max(mad(y - yhat, 1), 1e-6)));

    opt = optimset( ...
        'Display', 'off', ...
        'MaxFunEvals', 6000, ...
        'MaxIter', 6000);

    k0 = min(k_emp);

    % Fit P_R(k): exponential decay toward a plateau.
    PR0 = P_R(1);

    cR_lb = 0.00;
    cR_ub = 0.08;

    bnd_cR = @(c) min(max(c, cR_lb), cR_ub);

    pen_cR = @(c) ...
        1e6 * (c < cR_lb) .* (cR_lb - c).^2 + ...
        1e6 * (c > cR_ub) .* (c - cR_ub).^2;

    yhat_PR = @(p, x) ...
        bnd_cR(p(2)) + ...
        (PR0 - bnd_cR(p(2))) .* exp(-exp(p(1)) .* (x - k0));

    obj_PR = @(p) huber_loss(P_R, yhat_PR(p, k_emp)) + pen_cR(p(2));

    p_init_PR = [log(0.18); min(P_R(end), 0.05)];
    p_optR = fminsearch(obj_PR, p_init_PR, opt);

    bR_opt = exp(p_optR(1));
    cR_opt = bnd_cR(p_optR(2));

    PR_fit_emp = yhat_PR(p_optR, k_emp);
    PR_fit_full = yhat_PR(p_optR, k_vals);

    R2_PR = local_r2(P_R, PR_fit_emp);

    % Fit P_M(k): saturating exponential growth.
    PM0 = P_M(1);

    cM_lb = max(P_M);
    cM_ub = max(0.95, cM_lb + 1e-4);

    bnd_cM = @(c) min(max(c, cM_lb), cM_ub);

    pen_cM = @(c) ...
        1e6 * (c < cM_lb) .* (cM_lb - c).^2 + ...
        1e6 * (c > cM_ub) .* (c - cM_ub).^2;

    yhat_PM = @(p, x) ...
        bnd_cM(p(2)) - ...
        (bnd_cM(p(2)) - PM0) .* exp(-exp(p(1)) .* (x - k0));

    obj_PM = @(p) huber_loss(P_M, yhat_PM(p, k_emp)) + pen_cM(p(2));

    p_init_PM = [log(0.12); max(0.8, max(P_M))];
    p_optM = fminsearch(obj_PM, p_init_PM, opt);

    bM_opt = exp(p_optM(1));
    cM_opt = bnd_cM(p_optM(2));

    PM_fit_emp = yhat_PM(p_optM, k_emp);
    PM_fit_full = yhat_PM(p_optM, k_vals);

    R2_PM = local_r2(P_M, PM_fit_emp);

    % Fit mu_M(k): shrinkage toward zero.
    mask_muM = (k_emp >= 7) & ...
        (k_emp <= k_anchor) & ...
        isfinite(mu_M);

    if sum(mask_muM) < 3
        mask_muM = isfinite(mu_M);
    end

    xm = k_emp(mask_muM);
    ym = mu_M(mask_muM);

    if numel(xm) >= 3

        idx_anchor = find(k_emp == k_anchor, 1);
        muM_anchor = mu_M(idx_anchor);

        yhat_muM = @(p, x) ...
            muM_anchor .* exp(-exp(p(1)) .* (x - k_anchor));

        obj_muM = @(p) huber_loss(ym, yhat_muM(p, xm));

        p_opt_muM = fminsearch(obj_muM, log(0.08), opt);
        a_muM = exp(p_opt_muM(1));

        muM_fit_emp = yhat_muM(p_opt_muM, xm);
        muM_fit_full = muM_anchor .* exp(-a_muM .* (k_vals - k_anchor));

        R2_muM = local_r2(ym, muM_fit_emp);

    else

        a_muM = NaN;
        R2_muM = NaN;
        muM_fit_full = zeros(size(k_vals));
    end

    % Fit sigma_R(k): exponential decay.
    mask_sigR = (k_emp >= k_emp_min) & ...
        (k_emp <= k_anchor) & ...
        isfinite(sig_R) & ...
        (sig_R > 0);

    xs = k_emp(mask_sigR);
    ys = sig_R(mask_sigR);

    if numel(xs) >= 3

        k0_s = min(xs);

        c_lb_s = 0.0003;
        c_ub_s = 0.0012;

        bnd_cs = @(c) min(max(c, c_lb_s), c_ub_s);

        pen_cs = @(c) ...
            1e6 * (c < c_lb_s) .* (c_lb_s - c).^2 + ...
            1e6 * (c > c_ub_s) .* (c - c_ub_s).^2;

        yhat_sigR = @(p, x) ...
            bnd_cs(p(3)) + ...
            exp(p(1)) .* exp(-exp(p(2)) .* (x - k0_s));

        obj_sigR = @(p) huber_loss(ys, yhat_sigR(p, xs)) + pen_cs(p(3));

        init_amp = max(max(ys) - min(ys), 1e-5);
        p_init_sigR = [log(init_amp); log(0.18); min(ys)];

        p_opt_sigR = fminsearch(obj_sigR, p_init_sigR, opt);

        A_sigR = exp(p_opt_sigR(1));
        b_sigR = exp(p_opt_sigR(2));
        c_sigR = bnd_cs(p_opt_sigR(3));

        sigR_fit_emp = yhat_sigR(p_opt_sigR, xs);
        sigR_fit_full = yhat_sigR(p_opt_sigR, k_vals);

        R2_sigR = local_r2(ys, sigR_fit_emp);

    else

        A_sigR = NaN;
        b_sigR = NaN;
        c_sigR = NaN;
        R2_sigR = NaN;
        sigR_fit_full = NaN(size(k_vals));
    end

    % Construct residual distributions for k > 21.
    w_curve = min(max(PR_fit_full, 0), 0.40);

    if all(isnan(muM_fit_full))
        muM_curve = zeros(size(k_vals));
    else
        muM_curve = muM_fit_full;
    end

    idx_muR_ref = (k_emp >= 15) & (k_emp <= k_anchor) & isfinite(mu_R);

    if any(idx_muR_ref)
        muR_const = median(mu_R(idx_muR_ref), 'omitnan');
    else
        muR_const = mean(region_right);
    end

    muR_curve = muR_const * ones(size(k_vals));
    sigR_curve = max(sigR_fit_full, 1e-6);

    % Target RMSE curve:
    % Empirical all-inner RMSE values are kept for k = 5 to 21.
    % The fitted power-law model is used only for extrapolated k values.
    rmse_target = zeros(size(k_vals));

    for rr = 1:numel(k_vals)

        k_now = k_vals(rr);

        if k_now <= k_emp_max
            idx_tmp = find(k_emp == k_now, 1);
            rmse_target(rr) = rmseK(idx_tmp);
        else
            rmse_target(rr) = fit_RMSE(k_now);
        end
    end

    if any(~isfinite(rmse_target)) || any(rmse_target <= 0)
        error('Invalid RMSE target values were generated for scenario %s.', curr_name);
    end

    % Back-calculate the main-mode width sigma_M from the target RMSE.
    sigM_curve = zeros(size(k_vals));

    for rr = 1:numel(k_vals)

        rt = rmse_target(rr);
        w = w_curve(rr);

        muM0 = muM_curve(rr);
        muR0 = muR_curve(rr);
        sigR0 = sigR_curve(rr);

        w_eff = w;
        min_sigM = 0.0010;

        for it = 1:40

            num = rt^2 ...
                - w_eff * (sigR0^2 + muR0^2) ...
                - (1 - w_eff) * muM0^2;

            den = max(1 - w_eff, 1e-8);

            if (num / den) > min_sigM^2
                break;
            end

            w_eff = 0.92 * w_eff;
        end

        w_curve(rr) = w_eff;

        num = rt^2 ...
            - w_eff * (sigR0^2 + muR0^2) ...
            - (1 - w_eff) * muM0^2;

        sigM_curve(rr) = sqrt(max(num / max(1 - w_eff, 1e-8), min_sigM^2));
    end

    % Prepare common random numbers for extrapolated distributions.
    base_u = rand(N_sim_plot, 1);
    base_z = randn(N_sim_plot, 1);
    base_pick = rand(N_sim_plot, 1);

    err_anchor = err_cell{k_emp == k_anchor};

    DistSamples = cell(numel(plot_k_list), 1);
    DistType = strings(numel(plot_k_list), 1);

    for pp = 1:numel(plot_k_list)

        k_plot_now = plot_k_list(pp);

        if k_plot_now <= k_emp_max

            idx_emp_now = find(k_emp == k_plot_now, 1);

            DistSamples{pp} = err_cell{idx_emp_now};
            DistType(pp) = "empirical";

        else

            idx_val = find(k_vals == k_plot_now, 1);

            isR = base_pick < w_curve(idx_val);

            e_par = zeros(N_sim_plot, 1);

            e_par(~isR) = muM_curve(idx_val) + sigM_curve(idx_val) * base_z(~isR);
            e_par(isR)  = muR_curve(idx_val) + sigR_curve(idx_val) * base_z(isR);

            if k_plot_now < k_bridge_end

                t = (k_plot_now - k_anchor) / (k_bridge_end - k_anchor);
                t = min(max(t, 0), 1);
                alpha = 3 * t^2 - 2 * t^3;

                e_data = sort(err_anchor);
                p_emp = ((1:numel(e_data))' - 0.5) / numel(e_data);

                e_emp = interp1(p_emp, e_data, base_u, 'linear', 'extrap');

                e_sample = (1 - alpha) .* e_emp + alpha .* e_par;

                DistSamples{pp} = e_sample;
                DistType(pp) = "bridged";

            else

                DistSamples{pp} = e_par;
                DistType(pp) = "extrapolated";
            end
        end
    end

    % Check trend directions.
    p_slope_PM = polyfit(k_emp, P_M, 1);
    p_slope_PR = polyfit(k_emp, P_R, 1);

    if numel(xs) >= 3
        p_slope_sigR = polyfit(xs, ys, 1);
        slope_sigR = p_slope_sigR(1);
    else
        slope_sigR = NaN;
    end

    slope_PM = p_slope_PM(1);
    slope_PR = p_slope_PR(1);

    PM_change = P_M(end) - P_M(1);
    PR_change = P_R(end) - P_R(1);
    sigR_change = sig_R(end) - sig_R(1);

    % Build summary tables.
    FitSummary = table( ...
        curr_name, ...
        main_mult(ss), right_mult(ss), ...
        region_main(1), region_main(2), ...
        region_right(1), region_right(2), ...
        R2_PM, R2_PR, R2_muM, R2_sigR, R2_RMSE, ...
        slope_PM, slope_PR, slope_sigR, ...
        PM_change, PR_change, sigR_change, ...
        P_M(1), P_M(end), ...
        P_R(1), P_R(end), ...
        sig_R(1), sig_R(end), ...
        a_RMSE, b_RMSE, c_RMSE, ...
        'VariableNames', { ...
        'Scenario', ...
        'MainWidthMultiplier', 'RightWidthMultiplier', ...
        'MainLower', 'MainUpper', ...
        'RightLower', 'RightUpper', ...
        'R2_PM', 'R2_PR', 'R2_muM', 'R2_sigR', 'R2_RMSE', ...
        'Slope_PM', 'Slope_PR', 'Slope_sigR', ...
        'Delta_PM_k21_minus_k5', ...
        'Delta_PR_k21_minus_k5', ...
        'Delta_sigR_k21_minus_k5', ...
        'PM_k5', 'PM_k21', ...
        'PR_k5', 'PR_k21', ...
        'sigR_k5', 'sigR_k21', ...
        'a_RMSE', 'b_RMSE', 'c_RMSE'});

    FitSummary_All = [FitSummary_All; FitSummary]; %#ok<AGROW>

    FeatureTable = table( ...
        repmat(curr_name, num_k_emp, 1), ...
        k_emp, ...
        num_outer_used_list, ...
        num_inner_used_list, ...
        num_residual_used_list, ...
        repmat(region_main(1), num_k_emp, 1), ...
        repmat(region_main(2), num_k_emp, 1), ...
        repmat(region_right(1), num_k_emp, 1), ...
        repmat(region_right(2), num_k_emp, 1), ...
        P_M, P_R, P_V, P_L, ...
        mu_M, mu_R, sig_M, sig_R, rmseK, ...
        'VariableNames', { ...
        'Scenario', 'k', ...
        'N_outer_runs', ...
        'N_inner_runs_all_inner', ...
        'N_residuals_all_inner', ...
        'MainLower', 'MainUpper', ...
        'RightLower', 'RightUpper', ...
        'P_M', 'P_R', 'P_V', 'P_L', ...
        'mu_M', 'mu_R', 'sig_M', 'sig_R', 'RMSE'});

    FeatureCurve_All = [FeatureCurve_All; FeatureTable]; %#ok<AGROW>

    FitCurveTable = table( ...
        repmat(curr_name, numel(k_vals), 1), ...
        k_vals, ...
        PM_fit_full, ...
        PR_fit_full, ...
        muM_fit_full, ...
        sigR_fit_full, ...
        2 * sigR_fit_full, ...
        rmse_target, ...
        'VariableNames', { ...
        'Scenario', 'k', ...
        'PM_fit', 'PR_fit', 'muM_fit', ...
        'sigR_half_width_fit', ...
        'sigR_full_width_fit', ...
        'RMSE_target'});

    FitCurve_All = [FitCurve_All; FitCurveTable]; %#ok<AGROW>

    % Store scenario-specific results.
    Store(ss).Scenario = curr_name;
    Store(ss).k_emp = k_emp;
    Store(ss).k_vals = k_vals;

    Store(ss).P_M = P_M;
    Store(ss).P_R = P_R;
    Store(ss).sig_R = sig_R;

    Store(ss).PM_fit_full = PM_fit_full;
    Store(ss).PR_fit_full = PR_fit_full;
    Store(ss).sigR_fit_full = sigR_fit_full;
    Store(ss).sigR_full_width_fit = 2 * sigR_fit_full;
    Store(ss).RMSE_target = rmse_target;

    Store(ss).region_main = region_main;
    Store(ss).region_right = region_right;

    Store(ss).DistSamples = DistSamples;
    Store(ss).DistType = DistType;
end

%% 8) Save summary tables
writetable(FitSummary_All, cfg.fit_summary_csv);
writetable(FeatureCurve_All, cfg.feature_curves_csv);
writetable(FitCurve_All, cfg.fit_curves_csv);

disp('================ Region-width robustness fitting summary ================');
disp(FitSummary_All);

%% 9) Build and save trend-check table
TrendCheck = table();

for ss = 1:height(FitSummary_All)

    scen = FitSummary_All.Scenario(ss);

    PM_ok = FitSummary_All.Delta_PM_k21_minus_k5(ss) > 0;
    PR_ok = FitSummary_All.Delta_PR_k21_minus_k5(ss) < 0;
    sigR_ok = FitSummary_All.Delta_sigR_k21_minus_k5(ss) < 0;

    curr_row = table( ...
        scen, ...
        PM_ok, PR_ok, sigR_ok, ...
        FitSummary_All.Delta_PM_k21_minus_k5(ss), ...
        FitSummary_All.Delta_PR_k21_minus_k5(ss), ...
        FitSummary_All.Delta_sigR_k21_minus_k5(ss), ...
        'VariableNames', { ...
        'Scenario', ...
        'PM_increases_from_k5_to_k21', ...
        'PR_decreases_from_k5_to_k21', ...
        'sigR_decreases_from_k5_to_k21', ...
        'Delta_PM', 'Delta_PR', 'Delta_sigR'});

    TrendCheck = [TrendCheck; curr_row]; %#ok<AGROW>
end

disp(TrendCheck);

writetable(TrendCheck, cfg.trend_check_csv);

save(cfg.model_mat, ...
    'cfg', ...
    'ScenarioName', ...
    'main_mult', ...
    'right_mult', ...
    'k_emp', ...
    'k_vals', ...
    'plot_k_list', ...
    'AllInnerCountTable', ...
    'FitSummary_All', ...
    'FeatureCurve_All', ...
    'FitCurve_All', ...
    'TrendCheck', ...
    'Store', ...
    'a_RMSE', ...
    'b_RMSE', ...
    'c_RMSE', ...
    'R2_RMSE', ...
    '-v7.3');

fprintf('Region-width robustness tables saved to Output/Results.\n');
fprintf('Region-width robustness model saved: %s\n', cfg.model_mat);

%% 10) Supplementary Fig. 12a: R-squared summary plot
fig1 = figure( ...
    'Color', 'w', ...
    'Name', 'SupFig12a_ResidualRegionBoundary_R2_summary', ...
    'Units', 'centimeters', ...
    'Position', [2, 2, 18, 9]);

R2_mat = [
    FitSummary_All.R2_PM, ...
    FitSummary_All.R2_PR, ...
    FitSummary_All.R2_sigR
    ];

bar(R2_mat, 'grouped');
grid on;
box on;

ylim([0, 1.05]);

xticks(1:nScenario);
xticklabels(FitSummary_All.Scenario);
xtickangle(30);

ylabel('Goodness of fit, R^2', ...
    'FontName', 'Arial', ...
    'FontSize', 10);

legend({'P_M(k)', 'P_R(k)', '\sigma_R(k)'}, ...
    'Location', 'southoutside', ...
    'Orientation', 'horizontal', ...
    'FontName', 'Arial', ...
    'FontSize', 9);

title('Goodness-of-fit under residual-region perturbations', ...
    'FontName', 'Arial', ...
    'FontSize', 11, ...
    'FontWeight', 'normal');

set(gca, ...
    'FontName', 'Arial', ...
    'FontSize', 8, ...
    'LineWidth', 0.8, ...
    'TickLabelInterpreter', 'none');

local_save_figure(fig1, cfg.fig_r2_summary, cfg.png_r2_summary, cfg.png_resolution);

%% 11) Supplementary Fig. 12c: P_R(k) sensitivity to right-tail-region width
show_PM = ["Baseline", "Narrow_main_20", "Wide_main_20"];
legend_PM = ["Baseline", "Main region -20%", "Main region +20%"];

fig2 = figure( ...
    'Color', 'w', ...
    'Name', 'SupFig12b_ResidualRegionBoundary_PM_sensitivity', ...
    'Units', 'centimeters', ...
    'Position', [2, 2, 16, 10]);

hold on;
grid on;
box on;

color_PM = [
    0.00, 0.00, 0.00
    0.35, 0.20, 0.80
    0.20, 0.65, 0.20
    ];

line_style_PM = {'-', '--', '-.'};

for ii = 1:numel(show_PM)

    idx_show = find([Store.Scenario] == show_PM(ii), 1);

    plot(Store(idx_show).k_vals, Store(idx_show).PM_fit_full, ...
        line_style_PM{ii}, ...
        'Color', color_PM(ii, :), ...
        'LineWidth', 2.2, ...
        'DisplayName', legend_PM(ii));

    scatter(Store(idx_show).k_emp, Store(idx_show).P_M, ...
        28, ...
        color_PM(ii, :), ...
        'filled', ...
        'MarkerFaceAlpha', 0.60, ...
        'HandleVisibility', 'off');
end

xlabel('Number of labelled target packs, k', ...
    'FontName', 'Arial', ...
    'FontSize', 10);

ylabel('Main-region probability, P_M(k)', ...
    'FontName', 'Arial', ...
    'FontSize', 10);

title('Sensitivity of P_M(k) to main-region boundary width', ...
    'FontName', 'Arial', ...
    'FontSize', 11, ...
    'FontWeight', 'normal');

legend('Location', 'southeast', ...
    'FontName', 'Arial', ...
    'FontSize', 8);

set(gca, ...
    'FontName', 'Arial', ...
    'FontSize', 8, ...
    'LineWidth', 0.8);

local_save_figure(fig2, cfg.fig_PM, cfg.png_PM, cfg.png_resolution);

%% 12) Supplementary Fig. 12c: P_R(k) sensitivity to right-tail-region width
show_R = ["Baseline", "Narrow_right_20", "Wide_right_20"];
legend_R = ["Baseline", "Right-tail region -20%", "Right-tail region +20%"];

fig3 = figure( ...
    'Color', 'w', ...
    'Name', 'SupFig12c_ResidualRegionBoundary_PR_sensitivity', ...
    'Units', 'centimeters', ...
    'Position', [2, 2, 16, 10]);

hold on;
grid on;
box on;

color_R = [
    0.00, 0.00, 0.00
    0.75, 0.20, 0.20
    0.90, 0.50, 0.10
    ];

line_style_R = {'-', '--', '-.'};

for ii = 1:numel(show_R)

    idx_show = find([Store.Scenario] == show_R(ii), 1);

    plot(Store(idx_show).k_vals, Store(idx_show).PR_fit_full, ...
        line_style_R{ii}, ...
        'Color', color_R(ii, :), ...
        'LineWidth', 2.2, ...
        'DisplayName', legend_R(ii));

    scatter(Store(idx_show).k_emp, Store(idx_show).P_R, ...
        28, ...
        color_R(ii, :), ...
        'filled', ...
        'MarkerFaceAlpha', 0.60, ...
        'HandleVisibility', 'off');
end

xlabel('Number of labelled target packs, k', ...
    'FontName', 'Arial', ...
    'FontSize', 10);

ylabel('Right-tail probability, P_R(k)', ...
    'FontName', 'Arial', ...
    'FontSize', 10);

title('Sensitivity of P_R(k) to right-tail boundary width', ...
    'FontName', 'Arial', ...
    'FontSize', 11, ...
    'FontWeight', 'normal');

legend('Location', 'northeast', ...
    'FontName', 'Arial', ...
    'FontSize', 8);

set(gca, ...
    'FontName', 'Arial', ...
    'FontSize', 8, ...
    'LineWidth', 0.8);

local_save_figure(fig3, cfg.fig_PR, cfg.png_PR, cfg.png_resolution);

%% 13) Supplementary Fig. 12d: sigma_R(k) sensitivity to right-tail-region width
fig4 = figure( ...
    'Color', 'w', ...
    'Name', 'SupFig12d_ResidualRegionBoundary_SigmaR_sensitivity', ...
    'Units', 'centimeters', ...
    'Position', [2, 2, 16, 10]);

hold on;
grid on;
box on;

for ii = 1:numel(show_R)

    idx_show = find([Store.Scenario] == show_R(ii), 1);

    plot(Store(idx_show).k_vals, Store(idx_show).sigR_fit_full, ...
        line_style_R{ii}, ...
        'Color', color_R(ii, :), ...
        'LineWidth', 2.2, ...
        'DisplayName', legend_R(ii));

    scatter(Store(idx_show).k_emp, Store(idx_show).sig_R, ...
        28, ...
        color_R(ii, :), ...
        'filled', ...
        'MarkerFaceAlpha', 0.60, ...
        'HandleVisibility', 'off');
end

xlabel('Number of labelled target packs, k', ...
    'FontName', 'Arial', ...
    'FontSize', 10);

ylabel('Right-tail half-width, \sigma_R(k)', ...
    'FontName', 'Arial', ...
    'FontSize', 10);

title('Sensitivity of \sigma_R(k) to right-tail boundary width', ...
    'FontName', 'Arial', ...
    'FontSize', 11, ...
    'FontWeight', 'normal');

legend('Location', 'northeast', ...
    'FontName', 'Arial', ...
    'FontSize', 8);

set(gca, ...
    'FontName', 'Arial', ...
    'FontSize', 8, ...
    'LineWidth', 0.8);

local_save_figure(fig4, cfg.fig_sigR, cfg.png_sigR, cfg.png_resolution);

%% 14) Supplementary Fig. 12 residual-distribution plots for k = 5 to 25
if do_plot_distribution

    % Show three representative perturbation scenarios.
    show_dist_scenarios = ["Baseline", "Narrow_all_20", "Wide_all_20"];
    dist_scenario_labels = ["Baseline", "All regions -20%", "All regions +20%"];

    % Color settings.
    color_emp_hist = [0.82, 0.89, 0.96];
    color_emp_line = [0.10, 0.35, 0.75];

    color_ext_hist = [0.96, 0.84, 0.84];
    color_ext_line = [0.80, 0.20, 0.20];

    color_main_fill = [0.46, 0.73, 0.46];
    color_right_fill = [0.88, 0.55, 0.55];

    % Loop over each representative k value and each perturbation scenario.
    for kk = 1:numel(plot_k_list)

        k_now = plot_k_list(kk);

        for rr = 1:numel(show_dist_scenarios)

            scen_now = show_dist_scenarios(rr);
            scen_label = dist_scenario_labels(rr);

            idx_scen = find([Store.Scenario] == scen_now, 1);

            if isempty(idx_scen)
                warning('Scenario %s was not found in Store. Skipping.', scen_now);
                continue;
            end

            region_main = Store(idx_scen).region_main;
            region_right = Store(idx_scen).region_right;

            idx_k = find(plot_k_list == k_now, 1);

            e = Store(idx_scen).DistSamples{idx_k};
            type_now = Store(idx_scen).DistType(idx_k);

            if isempty(e) || all(~isfinite(e(:)))
                warning('No valid residual samples for scenario %s at k = %d. Skipping.', ...
                    scen_now, k_now);
                continue;
            end

            e = e(isfinite(e));

            fig_k = figure( ...
                'Color', 'w', ...
               'Name', sprintf('SupFig12_distribution_%s_k_%02d', scen_now, k_now), ...
                'Units', 'centimeters', ...
                'Position', [2, 2, 8, 6]);

            hold on;
            grid on;
            box on;

            if strcmp(type_now, "empirical")
                hist_color = color_emp_hist;
                line_color = color_emp_line;
            else
                hist_color = color_ext_hist;
                line_color = color_ext_line;
            end

            histogram(e, hist_edges, ...
                'Normalization', 'pdf', ...
                'FaceColor', hist_color, ...
                'EdgeColor', 'none', ...
                'FaceAlpha', 0.75);

            [f, xi] = ksdensity(e, 'Bandwidth', kde_bw);

            % Highlight the main residual region.
            idx_m = (xi >= region_main(1)) & (xi <= region_main(2));

            if any(idx_m)
                area(xi(idx_m), f(idx_m), ...
                    'FaceColor', color_main_fill, ...
                    'FaceAlpha', 0.35, ...
                    'EdgeColor', 'none');
            end

            % Highlight the right-tail residual region.
            idx_r = (xi >= region_right(1)) & (xi <= region_right(2));

            if any(idx_r)
                area(xi(idx_r), f(idx_r), ...
                    'FaceColor', color_right_fill, ...
                    'FaceAlpha', 0.35, ...
                    'EdgeColor', 'none');
            end

            plot(xi, f, '-', ...
                'Color', line_color, ...
                'LineWidth', 1.8);

            xline(0, 'k--', ...
                'LineWidth', 0.8);

            xline(region_main(1), '--', ...
                'Color', [0.20, 0.55, 0.20], ...
                'LineWidth', 0.8);

            xline(region_main(2), '--', ...
                'Color', [0.20, 0.55, 0.20], ...
                'LineWidth', 0.8);

            xline(region_right(1), '--', ...
                'Color', [0.70, 0.20, 0.20], ...
                'LineWidth', 0.8);

            xline(region_right(2), '--', ...
                'Color', [0.70, 0.20, 0.20], ...
                'LineWidth', 0.8);

            xlim([-0.06, 0.06]);

            yl = ylim;
            ylim([0, yl(2) * 1.10]);

            xlabel('Residual error', ...
                'FontName', 'Arial', ...
                'FontSize', 9);

            ylabel('Probability density', ...
                'FontName', 'Arial', ...
                'FontSize', 9);

            title(sprintf('%s, k = %d', scen_label, k_now), ...
                'FontName', 'Arial', ...
                'FontSize', 10, ...
                'FontWeight', 'normal');

            text(0.04, 0.90, char(type_now), ...
                'Units', 'normalized', ...
                'FontName', 'Arial', ...
                'FontSize', 8, ...
                'FontWeight', 'bold', ...
                'Color', line_color);

            set(gca, ...
                'FontName', 'Arial', ...
                'FontSize', 8, ...
                'LineWidth', 0.8, ...
                'TickDir', 'out');

            out_fig = fullfile(cfg.dist_dir, sprintf('SupFig12_distribution_%s_k_%02d.fig', scen_now, k_now));
            out_png = fullfile(cfg.dist_dir, sprintf('SupFig12_distribution_%s_k_%02d.png', scen_now, k_now));

            local_save_figure(fig_k, out_fig, out_png, cfg.png_resolution);

            close(fig_k);
        end
    end
end

fprintf('\n>>> P3 Exp04 residual-region boundary robustness completed.\n');
fprintf('>>> Saved fit summary:       %s\n', cfg.fit_summary_csv);
fprintf('>>> Saved feature curves:    %s\n', cfg.feature_curves_csv);
fprintf('>>> Saved fit curves:        %s\n', cfg.fit_curves_csv);
fprintf('>>> Saved trend check table: %s\n', cfg.trend_check_csv);
fprintf('>>> Saved all-inner counts:  %s\n', cfg.all_inner_counts_csv);
fprintf('>>> Saved model file:        %s\n', cfg.model_mat);
fprintf('>>> Saved figure folder: %s\n', cfg.supp_fig_dir);

fprintf('\n================ P3 Exp04 completed ================\n');

%% Local helper functions
function R2 = local_r2(y, yhat)
% Compute the coefficient of determination.

y = y(:);
yhat = yhat(:);

valid = isfinite(y) & isfinite(yhat);
y = y(valid);
yhat = yhat(valid);

if numel(y) < 2
    R2 = NaN;
    return;
end

den = sum((y - mean(y)).^2);

if den <= 0
    R2 = NaN;
else
    R2 = 1 - sum((y - yhat).^2) / den;
end
end

function local_save_figure(fig_handle, fig_file, png_file, resolution)
% Save a figure as both FIG and PNG files.

out_dir = fileparts(fig_file);

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

savefig(fig_handle, fig_file);
exportgraphics(fig_handle, png_file, 'Resolution', resolution);

fprintf('[Figure] Saved: %s\n', png_file);
end