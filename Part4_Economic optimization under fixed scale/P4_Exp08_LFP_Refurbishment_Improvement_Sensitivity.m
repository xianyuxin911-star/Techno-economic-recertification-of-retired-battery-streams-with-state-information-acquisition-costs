%% P4_Exp08_LFP_RefurbishmentImprovementSensitivity.m
% Purpose:
% This script evaluates the sensitivity of the fixed-scale LFP recertification model to 
% the refurbishment SOH improvement factor beta_refurbish and generates Extended Data Fig. 7.
%
% For each beta_refurbish scenario, k*, T1*, T2*, action shares, and cost
% decomposition are recalculated using the same k-dependent prediction-error
% model and threshold-grid search logic as the fixed-scale LFP
% analysis. The shared Part4 residual generator is used directly.


clear; clc; close all;

set(groot, 'DefaultFigureVisible', 'on');

%% 1) Configuration and path setup
cfg = struct();

script_dir = fileparts(mfilename('fullpath'));

if isempty(script_dir)
 script_dir = pwd;
end

cfg.part4_dir = script_dir;
cfg.output_dir = fullfile(cfg.part4_dir, 'Output');
cfg.results_dir = fullfile(cfg.output_dir, 'Results');
cfg.input_mat = fullfile(cfg.results_dir, 'P4_Exp01_FixedScale_LFP_Workspace.mat');
cfg.function_dir = fullfile(cfg.part4_dir, 'Function');

cfg.figure_root = fullfile(cfg.part4_dir, 'Figures');
cfg.extended_fig_dir = fullfile(cfg.figure_root, 'Extended');

cfg.summary_csv = fullfile(cfg.results_dir, 'P4_Exp08_LFP_RefurbishmentImprovementSensitivity_Summary.csv');
cfg.beta_csv = fullfile(cfg.results_dir, 'P4_Exp08_LFP_RefurbishmentImprovementSensitivity_BetaOnly.csv');
cfg.allK_csv = fullfile(cfg.results_dir, 'P4_Exp08_LFP_RefurbishmentImprovementSensitivity_AllK.csv');
cfg.result_mat = fullfile(cfg.results_dir, 'P4_Exp08_LFP_RefurbishmentImprovementSensitivity_Results.mat');
cfg.cache_mat = fullfile(cfg.results_dir, 'P4_Exp08_LFP_RefurbishmentImprovementSensitivity_PredictionCache.mat');

cfg.fig_beta_action_share = fullfile(cfg.extended_fig_dir, 'ExD07a_LFP_refurbishment_improvement_action_shares.fig');
cfg.png_beta_action_share = fullfile(cfg.extended_fig_dir, 'ExD07a_LFP_refurbishment_improvement_action_shares.png');
cfg.fig_beta_thresholds = fullfile(cfg.extended_fig_dir, 'ExD07b_LFP_refurbishment_improvement_thresholds.fig');
cfg.png_beta_thresholds = fullfile(cfg.extended_fig_dir, 'ExD07b_LFP_refurbishment_improvement_thresholds.png');
cfg.fig_beta_cost_decomp = fullfile(cfg.extended_fig_dir, 'ExD07c_LFP_refurbishment_improvement_cost_decomposition.fig');
cfg.png_beta_cost_decomp = fullfile(cfg.extended_fig_dir, 'ExD07c_LFP_refurbishment_improvement_cost_decomposition.png');

cfg.png_resolution = 600;

% Increment this value whenever the residual-generation logic changes.
cfg.cache_version = 2;

folder_list = {cfg.output_dir, cfg.results_dir, cfg.figure_root, cfg.extended_fig_dir};

for ii = 1:numel(folder_list)
 if ~exist(folder_list{ii}, 'dir')
 mkdir(folder_list{ii});
 end
end

addpath(cfg.function_dir);

if exist('local_build_noise_by_k', 'file') ~= 2
 error('Function local_build_noise_by_k.m was not found in: %s', cfg.function_dir);
end

fprintf('================ P4 Exp08: LFP refurbishment-improvement sensitivity ================\n');
fprintf('Input workspace file: %s\n', cfg.input_mat);
fprintf('Output folder: %s\n', cfg.results_dir);
fprintf('Figure folder: %s\n', cfg.extended_fig_dir);

%% 2) Load the fixed-scale LFP full workspace
mat_file = cfg.input_mat;

if ~isfile(mat_file)
 error(['Input file not found: %s\n' ...
 'Please run P4_Exp01_FixedScale_LFP_Optimization.m first to generate the fixed-scale LFP workspace.'], ...
 mat_file);
end

vars_in_file = who('-file', mat_file);

safe_var_list = { ...
 'k_vals', ...
 'T1_grid', ...
 'T2_grid', ...
 's_true', ...
 'Eta_1', ...
 'Eta_2', ...
 'Lambda_A', ...
 'Lambda_R', ...
 'Fixed_Scale_N', ...
 'c_pack', ...
 'c_train', ...
 'k_emp', ...
 'k_anchor', ...
 'k_bridge_end', ...
 'err_cell', ...
 'w_curve', ...
 'muM_curve', ...
 'sigM_curve', ...
 'muR_curve', ...
 'sigR_curve', ...
 'rmse_target', ...
 'base_u', ...
 'base_z', ...
 'base_pick', ...
 'base_bridge', ...
 'calc_V_reuse_base', ...
 'calc_V_recycle', ...
 'beta_refurbish', ...
 'C0', ...
 'C1', ...
 'gamma', ...
 'C_A1_extra', ...
 'C_A2_extra'};

vars_to_load = intersect(safe_var_list, vars_in_file, 'stable');

S = load(mat_file, vars_to_load{:});

close all;

required_vars = { ...
 'k_vals', ...
 'T1_grid', ...
 'T2_grid', ...
 's_true', ...
 'Eta_1', ...
 'Eta_2', ...
 'Lambda_A', ...
 'Lambda_R', ...
 'Fixed_Scale_N', ...
 'c_pack', ...
 'c_train', ...
 'k_emp', ...
 'k_anchor', ...
 'k_bridge_end', ...
 'err_cell', ...
 'w_curve', ...
 'muM_curve', ...
 'sigM_curve', ...
 'muR_curve', ...
 'sigR_curve', ...
 'rmse_target', ...
 'base_u', ...
 'base_z', ...
 'base_pick', ...
 'base_bridge', ...
 'calc_V_reuse_base', ...
 'calc_V_recycle'};

for ii = 1:numel(required_vars)
 if ~isfield(S, required_vars{ii})
 error('Required variable is missing from %s: %s', ...
 mat_file, required_vars{ii});
 end
end

k_vals = S.k_vals(:);
T1_grid = S.T1_grid(:);
T2_grid = S.T2_grid(:);

s_true = S.s_true(:);

Eta_1 = S.Eta_1;
Eta_2 = S.Eta_2;
Lambda_A = S.Lambda_A;
Lambda_R = S.Lambda_R;

Fixed_Scale_N = S.Fixed_Scale_N;
c_pack = S.c_pack;
c_train = S.c_train;

calc_V_reuse_base = S.calc_V_reuse_base;
calc_V_recycle = S.calc_V_recycle;

if isfield(S, 'beta_refurbish')
 beta_refurbish_base = S.beta_refurbish;
else
 beta_refurbish_base = 0.09;
end

if isfield(S, 'C0')
 C0 = S.C0;
else
 C0 = 92;
end

if isfield(S, 'C1')
 C1 = S.C1;
else
 C1 = 65;
end

if isfield(S, 'gamma')
 gamma = S.gamma;
else
 gamma = 3;
end

if isfield(S, 'C_A1_extra')
 C_A1_extra = S.C_A1_extra;
else
 C_A1_extra = 10;
end

if isfield(S, 'C_A2_extra')
 C_A2_extra = S.C_A2_extra;
else
 C_A2_extra = 15;
end

k_emp = S.k_emp(:);
k_anchor = S.k_anchor;
k_bridge_end = S.k_bridge_end;
err_cell = S.err_cell;

w_curve = S.w_curve(:);
muM_curve = S.muM_curve(:);
sigM_curve = S.sigM_curve(:);
muR_curve = S.muR_curve(:);
sigR_curve = S.sigR_curve(:);
rmse_target = S.rmse_target(:);

base_u = S.base_u(:);
base_z = S.base_z(:);
base_pick = S.base_pick(:);
base_bridge = S.base_bridge(:);

N_sim = numel(s_true);
num_k = numel(k_vals);
num_T1 = numel(T1_grid);
num_T2 = numel(T2_grid);

if numel(base_u) ~= N_sim || numel(base_z) ~= N_sim || ...
 numel(base_pick) ~= N_sim || numel(base_bridge) ~= N_sim
 error('base_u, base_z, base_pick, and base_bridge must have length N_sim.');
end

% Cache signature prevents reuse after the Part3 residual model, common
% random numbers, or simulated SOH population changes.
cache_signature = [ ...
 cfg.cache_version; ...
 N_sim; ...
 sum(s_true); sum(s_true.^2); ...
 sum(base_u); sum(base_u.^2); ...
 sum(base_z); sum(base_z.^2); ...
 sum(base_pick); sum(base_pick.^2); ...
 sum(base_bridge); sum(base_bridge.^2); ...
 sum(rmse_target); sum(w_curve); ...
 sum(muM_curve); sum(sigM_curve); ...
 sum(muR_curve); sum(sigR_curve)];

fprintf('Fixed-scale LFP workspace loaded.\n');
fprintf('k range = %d to %d, with %d k values.\n', ...
 min(k_vals), max(k_vals), num_k);
fprintf('T1 grid: %.3f to %.3f, with %d grid points.\n', ...
 min(T1_grid), max(T1_grid), num_T1);
fprintf('T2 grid: %.3f to %.3f, with %d grid points.\n', ...
 min(T2_grid), max(T2_grid), num_T2);
fprintf('Baseline beta_refurbish = %.4f\n', beta_refurbish_base);
fprintf('Baseline refurbishment cost: C0 = %.4f, C1 = %.4f, gamma = %.4f\n', ...
 C0, C1, gamma);
fprintf('C_A1_extra = %.4f, C_A2_extra = %.4f\n', ...
 C_A1_extra, C_A2_extra);

%% 3) Define refurbishment-improvement sensitivity range
beta_refurbish_list = (0.04:0.01:0.25)';

%% 4) Construct refurbishment-improvement sensitivity scenarios
Scenario_Name = {};
Scenario_Type = {};
beta_refurbish_scn = [];
refurb_cost_multiplier_scn = [];

for ii = 1:numel(beta_refurbish_list)

 Scenario_Name{end+1, 1} = sprintf('beta_%g', beta_refurbish_list(ii));
 Scenario_Type{end+1, 1} = 'Refurbishment improvement sensitivity';

 beta_refurbish_scn(end+1, 1) = beta_refurbish_list(ii);
 refurb_cost_multiplier_scn(end+1, 1) = 1.0;
end

Scenario_Table = table( ...
 Scenario_Name, ...
 Scenario_Type, ...
 beta_refurbish_scn, ...
 refurb_cost_multiplier_scn, ...
 'VariableNames', { ...
 'Scenario', ...
 'Scenario_type', ...
 'beta_refurbish', ...
 'refurb_cost_multiplier'});

n_scn = height(Scenario_Table);

disp(' ');
disp('========== Extended Data Fig. 7 refurbishment-improvement scenarios ==========');
disp(Scenario_Table);

%% 5) Precompute sorted predicted SOH and threshold positions for each k
use_cache = false;

if isfile(cfg.cache_mat)

 C = load(cfg.cache_mat);

 has_required_cache = isfield(C, 'SortOrder_byK') && ...
 isfield(C, 'idxT1_byK') && ...
 isfield(C, 'idxT2_byK') && ...
 isfield(C, 'k_vals') && ...
 isfield(C, 'T1_grid') && ...
 isfield(C, 'T2_grid') && ...
 isfield(C, 'N_sim') && ...
 isfield(C, 'cache_version') && ...
 isfield(C, 'cache_signature');

 if has_required_cache

 cache_size_ok = isequal(size(C.SortOrder_byK), [N_sim, num_k]) && ...
 isequal(size(C.idxT1_byK), [num_k, num_T1]) && ...
 isequal(size(C.idxT2_byK), [num_k, num_T2]);

 cache_grid_ok = isequal(C.k_vals(:), k_vals(:)) && ...
 isequal(C.T1_grid(:), T1_grid(:)) && ...
 isequal(C.T2_grid(:), T2_grid(:)) && ...
 C.N_sim == N_sim;

 cache_model_ok = isequal(C.cache_version, cfg.cache_version) && ...
 isequaln(C.cache_signature(:), cache_signature(:));

 if cache_size_ok && cache_grid_ok && cache_model_ok
 use_cache = true;
 end
 end
end

if use_cache

 fprintf('\nValid sorted-prediction cache detected: %s\n', cfg.cache_mat);

 SortOrder_byK = C.SortOrder_byK;
 idxT1_byK = C.idxT1_byK;
 idxT2_byK = C.idxT2_byK;

else

 fprintf('\nNo valid sorted-prediction cache detected. Precomputing sorted predicted SOH for each k.\n');

 SortOrder_byK = zeros(N_sim, num_k, 'uint32');
 idxT1_byK = zeros(num_k, num_T1);
 idxT2_byK = zeros(num_k, num_T2);

 for kk = 1:num_k

 k_current = k_vals(kk);

 sampled_noise = local_build_noise_by_k( ...
 k_current, ...
 k_emp, k_anchor, k_bridge_end, ...
 err_cell, ...
 w_curve, muM_curve, sigM_curve, ...
 muR_curve, sigR_curve, rmse_target, k_vals, ...
 base_u, base_z, base_pick, base_bridge);

 s_pred = min(max(s_true + sampled_noise, 0), 1.0);

 [s_pred_sorted, order] = sort(s_pred, 'ascend');

 SortOrder_byK(:, kk) = uint32(order);

 for i_t1 = 1:num_T1
 idxT1_byK(kk, i_t1) = sum(s_pred_sorted < T1_grid(i_t1));
 end

 for i_t2 = 1:num_T2
 idxT2_byK(kk, i_t2) = sum(s_pred_sorted < T2_grid(i_t2));
 end

 fprintf(' k = %d sorted (%d/%d)\n', ...
 k_current, kk, num_k);
 end

 cache_version = cfg.cache_version;

 save(cfg.cache_mat, ...
 'SortOrder_byK', ...
 'idxT1_byK', ...
 'idxT2_byK', ...
 'k_vals', ...
 'T1_grid', ...
 'T2_grid', ...
 'N_sim', ...
 'cache_version', ...
 'cache_signature', ...
 '-v7.3');

 fprintf('Sorted-prediction cache saved: %s\n', cfg.cache_mat);
end

%% 6) Recalculate A2 value and optimize k*, T1*, and T2* for each beta scenario
k_star = zeros(n_scn, 1);
T1_star = zeros(n_scn, 1);
T2_star = zeros(n_scn, 1);

A1_share = zeros(n_scn, 1);
A2_share = zeros(n_scn, 1);
A3_share = zeros(n_scn, 1);

True_A1_share = zeros(n_scn, 1);
True_A2_share = zeros(n_scn, 1);
True_A3_share = zeros(n_scn, 1);

Opportunity_loss_USD = zeros(n_scn, 1);
Verification_cost_USD = zeros(n_scn, 1);
Risk_penalty_USD = zeros(n_scn, 1);
Information_cost_USD = zeros(n_scn, 1);
Total_cost_USD = zeros(n_scn, 1);
Total_cost_per_pack_USD = zeros(n_scn, 1);

TotalCost_byScenario_K = zeros(n_scn, num_k);
O_byScenario_K = zeros(n_scn, num_k);
A_byScenario_K = zeros(n_scn, num_k);
R_byScenario_K = zeros(n_scn, num_k);
Info_byScenario_K = zeros(n_scn, num_k);
T1_byScenario_K = zeros(n_scn, num_k);
T2_byScenario_K = zeros(n_scn, num_k);
A1share_byScenario_K = zeros(n_scn, num_k);
A2share_byScenario_K = zeros(n_scn, num_k);
A3share_byScenario_K = zeros(n_scn, num_k);

fprintf('\nStarting refurbishment-improvement scenario optimization.\n');

for ss = 1:n_scn

 beta_now = Scenario_Table.beta_refurbish(ss);
 mult_now = Scenario_Table.refurb_cost_multiplier(ss);

 C_refurbish_now = @(soh) mult_now .* (C0 + C1 .* (1 - soh).^gamma);

 V_A1 = calc_V_reuse_base(s_true) - C_A1_extra;

 V_A2 = calc_V_reuse_base(min(s_true + beta_now .* (1 - s_true), 1.0)) ...
 - C_refurbish_now(s_true) ...
 - C_A1_extra ...
 - C_A2_extra;

 V_A3 = calc_V_recycle(s_true);

 Pi_True_now = [V_A1(:), V_A2(:), V_A3(:)];
 [Pi_True_Max_now, True_Action_i_now] = max(Pi_True_now, [], 2);

 True_A1_share(ss) = mean(True_Action_i_now == 1);
 True_A2_share(ss) = mean(True_Action_i_now == 2);
 True_A3_share(ss) = mean(True_Action_i_now == 3);

 O_action = zeros(N_sim, 3);
 A_action = zeros(N_sim, 3);
 R_action = zeros(N_sim, 3);

 for a_pred = 1:3

 O_arr = Pi_True_Max_now - Pi_True_now(:, a_pred);

 is_G1_error = (True_Action_i_now == 3 & a_pred == 2) | ...
 (True_Action_i_now == 2 & a_pred == 3);

 is_G2_error = (True_Action_i_now == 1 & a_pred == 2) | ...
 (True_Action_i_now == 2 & a_pred == 1);

 Current_Eta = zeros(N_sim, 1);
 Current_Eta(is_G1_error) = Eta_1;
 Current_Eta(is_G2_error) = Eta_2;

 trigger_penalty = O_arr > Current_Eta;
 Excess_Loss = max(0, O_arr - Current_Eta);

 A_arr = trigger_penalty .* Lambda_A;
 R_arr = (a_pred < True_Action_i_now) .* trigger_penalty .* ...
 (Lambda_R .* Excess_Loss);

 O_action(:, a_pred) = O_arr;
 A_action(:, a_pred) = A_arr;
 R_action(:, a_pred) = R_arr;
 end

 O_best_k = zeros(num_k, 1);
 A_best_k = zeros(num_k, 1);
 R_best_k = zeros(num_k, 1);
 EL_best_k = zeros(num_k, 1);
 T1_best_k = zeros(num_k, 1);
 T2_best_k = zeros(num_k, 1);
 A1_share_k = zeros(num_k, 1);
 A2_share_k = zeros(num_k, 1);
 A3_share_k = zeros(num_k, 1);

 for kk = 1:num_k

 order = double(SortOrder_byK(:, kk));

 O1 = O_action(order, 1);
 O2 = O_action(order, 2);
 O3 = O_action(order, 3);

 A1c = A_action(order, 1);
 A2c = A_action(order, 2);
 A3c = A_action(order, 3);

 R1 = R_action(order, 1);
 R2 = R_action(order, 2);
 R3 = R_action(order, 3);

 L1 = O1 + A1c + R1;
 L2 = O2 + A2c + R2;
 L3 = O3 + A3c + R3;

 pL1 = [0; cumsum(L1)];
 pL2 = [0; cumsum(L2)];
 pL3 = [0; cumsum(L3)];

 pO1 = [0; cumsum(O1)];
 pO2 = [0; cumsum(O2)];
 pO3 = [0; cumsum(O3)];

 pA1 = [0; cumsum(A1c)];
 pA2 = [0; cumsum(A2c)];
 pA3 = [0; cumsum(A3c)];

 pR1 = [0; cumsum(R1)];
 pR2 = [0; cumsum(R2)];
 pR3 = [0; cumsum(R3)];

 min_loss_sum = inf;
 best_i_t1 = NaN;
 best_i_t2 = NaN;

 for i_t1 = 1:num_T1

 T1_now = T1_grid(i_t1);
 idx1 = idxT1_byK(kk, i_t1);

 for i_t2 = 1:num_T2

 T2_now = T2_grid(i_t2);

 if T1_now >= T2_now
 continue;
 end

 idx2 = idxT2_byK(kk, i_t2);

 loss_sum = ...
 pL3(idx1 + 1) + ...
 (pL2(idx2 + 1) - pL2(idx1 + 1)) + ...
 (pL1(end) - pL1(idx2 + 1));

 if loss_sum < min_loss_sum

 min_loss_sum = loss_sum;
 best_i_t1 = i_t1;
 best_i_t2 = i_t2;
 end
 end
 end

 idx1_best = idxT1_byK(kk, best_i_t1);
 idx2_best = idxT2_byK(kk, best_i_t2);

 O_sum = ...
 pO3(idx1_best + 1) + ...
 (pO2(idx2_best + 1) - pO2(idx1_best + 1)) + ...
 (pO1(end) - pO1(idx2_best + 1));

 A_sum = ...
 pA3(idx1_best + 1) + ...
 (pA2(idx2_best + 1) - pA2(idx1_best + 1)) + ...
 (pA1(end) - pA1(idx2_best + 1));

 R_sum = ...
 pR3(idx1_best + 1) + ...
 (pR2(idx2_best + 1) - pR2(idx1_best + 1)) + ...
 (pR1(end) - pR1(idx2_best + 1));

 O_best_k(kk) = O_sum / N_sim;
 A_best_k(kk) = A_sum / N_sim;
 R_best_k(kk) = R_sum / N_sim;
 EL_best_k(kk) = min_loss_sum / N_sim;

 T1_best_k(kk) = T1_grid(best_i_t1);
 T2_best_k(kk) = T2_grid(best_i_t2);

 A3_share_k(kk) = idx1_best / N_sim;
 A2_share_k(kk) = (idx2_best - idx1_best) / N_sim;
 A1_share_k(kk) = (N_sim - idx2_best) / N_sim;
 end

 Information_cost_byK = k_vals .* c_pack + c_train;
 Total_cost_byK = Fixed_Scale_N .* EL_best_k + Information_cost_byK;

 [Total_min, idx_k_min] = min(Total_cost_byK);

 k_star(ss) = k_vals(idx_k_min);
 T1_star(ss) = T1_best_k(idx_k_min);
 T2_star(ss) = T2_best_k(idx_k_min);

 Opportunity_loss_USD(ss) = Fixed_Scale_N .* O_best_k(idx_k_min);
 Verification_cost_USD(ss) = Fixed_Scale_N .* A_best_k(idx_k_min);
 Risk_penalty_USD(ss) = Fixed_Scale_N .* R_best_k(idx_k_min);
 Information_cost_USD(ss) = Information_cost_byK(idx_k_min);

 Total_cost_USD(ss) = Total_min;
 Total_cost_per_pack_USD(ss) = Total_min ./ Fixed_Scale_N;

 A1_share(ss) = A1_share_k(idx_k_min);
 A2_share(ss) = A2_share_k(idx_k_min);
 A3_share(ss) = A3_share_k(idx_k_min);

 TotalCost_byScenario_K(ss, :) = Total_cost_byK(:)';
 O_byScenario_K(ss, :) = Fixed_Scale_N .* O_best_k(:)';
 A_byScenario_K(ss, :) = Fixed_Scale_N .* A_best_k(:)';
 R_byScenario_K(ss, :) = Fixed_Scale_N .* R_best_k(:)';
 Info_byScenario_K(ss, :) = Information_cost_byK(:)';
 T1_byScenario_K(ss, :) = T1_best_k(:)';
 T2_byScenario_K(ss, :) = T2_best_k(:)';
 A1share_byScenario_K(ss, :) = A1_share_k(:)';
 A2share_byScenario_K(ss, :) = A2_share_k(:)';
 A3share_byScenario_K(ss, :) = A3_share_k(:)';

 fprintf([' %-14s | beta = %.3f | ' ...
 'k* = %3d, T1 = %.3f, T2 = %.3f, ' ...
 'A1 = %.3f, A2 = %.3f, A3 = %.3f, cost = %.4f USD/pack\n'], ...
 Scenario_Table.Scenario{ss}, ...
 beta_now, ...
 k_star(ss), T1_star(ss), T2_star(ss), ...
 A1_share(ss), A2_share(ss), A3_share(ss), ...
 Total_cost_per_pack_USD(ss));
end

fprintf('Refurbishment-improvement scenario optimization completed.\n');

%% 7) Build and save result tables
Result_Table = [Scenario_Table, table( ...
 k_star, ...
 T1_star, ...
 T2_star, ...
 True_A1_share, ...
 True_A2_share, ...
 True_A3_share, ...
 A1_share, ...
 A2_share, ...
 A3_share, ...
 Opportunity_loss_USD, ...
 Verification_cost_USD, ...
 Risk_penalty_USD, ...
 Information_cost_USD, ...
 Total_cost_USD, ...
 Total_cost_per_pack_USD, ...
 'VariableNames', { ...
 'k_star', ...
 'T1_star', ...
 'T2_star', ...
 'True_A1_share', ...
 'True_A2_share', ...
 'True_A3_share', ...
 'Pred_A1_share', ...
 'Pred_A2_share', ...
 'Pred_A3_share', ...
 'Opportunity_loss_USD', ...
 'Verification_cost_USD', ...
 'Risk_penalty_USD', ...
 'Information_cost_USD', ...
 'Total_cost_USD', ...
 'Total_cost_per_pack_USD'})];

disp(' ');
disp('========== Extended Data Fig. 7 refurbishment-improvement sensitivity summary ==========');
disp(Result_Table);

writetable(Result_Table, cfg.summary_csv);

Result_beta = Result_Table(strcmp(Result_Table.Scenario_type, ...
 'Refurbishment improvement sensitivity'), :);

Result_beta = sortrows(Result_beta, 'beta_refurbish');

writetable(Result_beta, cfg.beta_csv);

AllK_Table = table(k_vals, 'VariableNames', {'k'});

for ss = 1:n_scn

 name_now = matlab.lang.makeValidName(Result_Table.Scenario{ss});

 AllK_Table.(['TotalCost_' name_now '_USD']) = TotalCost_byScenario_K(ss, :)';
 AllK_Table.(['Opportunity_' name_now '_USD']) = O_byScenario_K(ss, :)';
 AllK_Table.(['Verification_' name_now '_USD']) = A_byScenario_K(ss, :)';
 AllK_Table.(['Risk_' name_now '_USD']) = R_byScenario_K(ss, :)';
 AllK_Table.(['Information_' name_now '_USD']) = Info_byScenario_K(ss, :)';
 AllK_Table.(['T1_' name_now]) = T1_byScenario_K(ss, :)';
 AllK_Table.(['T2_' name_now]) = T2_byScenario_K(ss, :)';
 AllK_Table.(['A1share_' name_now]) = A1share_byScenario_K(ss, :)';
 AllK_Table.(['A2share_' name_now]) = A2share_byScenario_K(ss, :)';
 AllK_Table.(['A3share_' name_now]) = A3share_byScenario_K(ss, :)';
end

writetable(AllK_Table, cfg.allK_csv);

save(cfg.result_mat, ...
 'cfg', ...
 'Scenario_Table', ...
 'Result_Table', ...
 'Result_beta', ...
 'AllK_Table', ...
 'beta_refurbish_list', ...
 'beta_refurbish_base', ...
 'C0', ...
 'C1', ...
 'gamma', ...
 'C_A1_extra', ...
 'C_A2_extra', ...
 'Eta_1', ...
 'Eta_2', ...
 'Lambda_A', ...
 'Lambda_R', ...
 'Fixed_Scale_N', ...
 'c_pack', ...
 'c_train', ...
 'k_vals', ...
 'T1_grid', ...
 'T2_grid', ...
 'TotalCost_byScenario_K', ...
 'O_byScenario_K', ...
 'A_byScenario_K', ...
 'R_byScenario_K', ...
 'Info_byScenario_K', ...
 'T1_byScenario_K', ...
 'T2_byScenario_K', ...
 'A1share_byScenario_K', ...
 'A2share_byScenario_K', ...
 'A3share_byScenario_K', ...
 'k_star', ...
 'T1_star', ...
 'T2_star', ...
 'True_A1_share', ...
 'True_A2_share', ...
 'True_A3_share', ...
 'A1_share', ...
 'A2_share', ...
 'A3_share', ...
 'Opportunity_loss_USD', ...
 'Verification_cost_USD', ...
 'Risk_penalty_USD', ...
 'Information_cost_USD', ...
 'Total_cost_USD', ...
 'Total_cost_per_pack_USD', ...
 '-v7.3');

fprintf('Summary table saved: %s\n', cfg.summary_csv);
fprintf('Beta-only table saved: %s\n', cfg.beta_csv);
fprintf('All-k table saved: %s\n', cfg.allK_csv);
fprintf('MAT result file saved: %s\n', cfg.result_mat);

%% 8) Generate and save figures

% Extended Data Fig. 7a: beta_refurbish vs action shares.
fig1 = figure( ...
 'Name', 'ExD07a_LFP_refurbishment_improvement_action_shares', ...
 'Color', 'w', ...
 'Position', [100, 80, 760, 560]);

x = Result_beta.beta_refurbish;

h1 = plot(x, Result_beta.Pred_A1_share, '-o', ...
 'LineWidth', 2.0, ...
 'MarkerSize', 7, ...
 'MarkerFaceColor', 'w');

hold on;

h2 = plot(x, Result_beta.Pred_A2_share, '-s', ...
 'LineWidth', 2.0, ...
 'MarkerSize', 7, ...
 'MarkerFaceColor', 'w');

h3 = plot(x, Result_beta.Pred_A3_share, '-^', ...
 'LineWidth', 2.0, ...
 'MarkerSize', 7, ...
 'MarkerFaceColor', 'w');

xline(beta_refurbish_base, '--', 'Baseline', ...
 'LineWidth', 1.2, ...
 'LabelVerticalAlignment', 'bottom', ...
 'HandleVisibility', 'off');

grid on;
box on;

xlabel('\beta_{ref}');
ylabel('Predicted action share');
title('Action-allocation sensitivity to refurbishment improvement');

legend([h1, h2, h3], ...
 {'A_1 direct reuse', 'A_2 refurbishment', 'A_3 recycling'}, ...
 'Location', 'best', ...
 'Box', 'off');

ylim([0, 1]);

set(gca, ...
 'FontName', 'Arial', ...
 'FontSize', 12, ...
 'LineWidth', 1.0);

local_save_figure(fig1, cfg.fig_beta_action_share, cfg.png_beta_action_share, cfg.png_resolution);

% Extended Data Fig. 7b: beta_refurbish vs optimized thresholds.
fig2 = figure( ...
 'Name', 'ExD07b_LFP_refurbishment_improvement_thresholds', ...
 'Color', 'w', ...
 'Position', [140, 100, 760, 560]);

x = Result_beta.beta_refurbish;

yyaxis left

h1 = plot(x, Result_beta.T1_star, '-o', ...
 'LineWidth', 2.0, ...
 'MarkerSize', 7, ...
 'MarkerFaceColor', 'w');

ylabel('Optimal SOH threshold, T_1^*');

ymin1 = min(Result_beta.T1_star) - 0.01;
ymax1 = max(Result_beta.T1_star) + 0.01;

if abs(ymax1 - ymin1) < 1e-6
 ymin1 = Result_beta.T1_star(1) - 0.01;
 ymax1 = Result_beta.T1_star(1) + 0.01;
end

ylim([ymin1, ymax1]);

yyaxis right

h2 = plot(x, Result_beta.T2_star, '-s', ...
 'LineWidth', 2.0, ...
 'MarkerSize', 7, ...
 'MarkerFaceColor', 'w');

ylabel('Optimal SOH threshold, T_2^*');

ymin2 = min(Result_beta.T2_star) - 0.01;
ymax2 = max(Result_beta.T2_star) + 0.01;

if abs(ymax2 - ymin2) < 1e-6
 ymin2 = Result_beta.T2_star(1) - 0.01;
 ymax2 = Result_beta.T2_star(1) + 0.01;
end

ylim([ymin2, ymax2]);

xline(beta_refurbish_base, '--', 'Baseline', ...
 'LineWidth', 1.2, ...
 'LabelVerticalAlignment', 'bottom', ...
 'HandleVisibility', 'off');

grid on;
box on;

xlabel('\beta_{ref}');
title('Optimized thresholds under refurbishment-improvement sensitivity');

legend([h1, h2], {'T_1^*', 'T_2^*'}, ...
 'Location', 'best', ...
 'Box', 'off');

set(gca, ...
 'FontName', 'Arial', ...
 'FontSize', 12, ...
 'LineWidth', 1.0);

local_save_figure(fig2, cfg.fig_beta_thresholds, cfg.png_beta_thresholds, cfg.png_resolution);

% Extended Data Fig. 7c: cost decomposition under beta_refurbish sensitivity.
fig3 = figure( ...
 'Name', 'ExD07c_LFP_refurbishment_improvement_cost_decomposition', ...
 'Color', 'w', ...
 'Position', [180, 120, 840, 560]);

x = Result_beta.beta_refurbish;

CostStack = [ ...
 Result_beta.Opportunity_loss_USD, ...
 Result_beta.Verification_cost_USD, ...
 Result_beta.Risk_penalty_USD, ...
 Result_beta.Information_cost_USD] ./ 1e4;

b = bar(x, CostStack, 'stacked', ...
 'EdgeColor', 'none', ...
 'BarWidth', 0.85);

hold on;

hTotal = plot(x, Result_beta.Total_cost_USD ./ 1e4, '-k', ...
 'LineWidth', 2.4, ...
 'Marker', 'o', ...
 'MarkerSize', 6, ...
 'MarkerFaceColor', 'w');

xline(beta_refurbish_base, '--', 'Baseline', ...
 'LineWidth', 1.2, ...
 'LabelVerticalAlignment', 'bottom', ...
 'HandleVisibility', 'off');

grid on;
box on;

xlabel('\beta_{ref}');
ylabel('Cost (10^4 USD)');
title('Cost decomposition under refurbishment-improvement sensitivity');

legend([b(1), b(2), b(3), b(4), hTotal], ...
 {'Opportunity loss', ...
 'Verification cost', ...
 'Risk penalty', ...
 'Information cost', ...
 'Total cost'}, ...
 'Location', 'northwest', ...
 'Box', 'off');

set(gca, ...
 'FontName', 'Arial', ...
 'FontSize', 12, ...
 'LineWidth', 1.0);

local_save_figure(fig3, cfg.fig_beta_cost_decomp, cfg.png_beta_cost_decomp, cfg.png_resolution);

drawnow;

fprintf('\n>>> Extended Data Fig. 7 LFP refurbishment-improvement sensitivity completed.\n');
fprintf('>>> Summary table saved: %s\n', cfg.summary_csv);
fprintf('>>> Beta-only table saved: %s\n', cfg.beta_csv);
fprintf('>>> All-k table saved: %s\n', cfg.allK_csv);
fprintf('>>> MAT result file saved: %s\n', cfg.result_mat);
fprintf('>>> Figures saved in: %s\n', cfg.extended_fig_dir);
fprintf('=========================================================================================\n');

%% Local helper function
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