%% P4_Exp06_TrueSOH_ActionValueBenchmark.m
% Purpose:
% This script computes the true-SOH action-value benchmark for LFP and NMC
%  recertification pathways and generates Extended Data Fig. 4.

clear; clc; close all;

set(groot, 'DefaultFigureVisible', 'on');

%% 1) Configuration and path setup
cfg = struct();

script_dir = fileparts(mfilename('fullpath'));

if isempty(script_dir)
    script_dir = pwd;
end

cfg.part4_dir = script_dir;
cfg.input_dir = fullfile(cfg.part4_dir, 'Input');

cfg.lfp_soh_mat = fullfile(cfg.input_dir, 'LFP_SOH_Table.mat');
cfg.nmc_soh_mat = fullfile(cfg.input_dir, 'NMC_SOH_Table.mat');

cfg.output_dir = fullfile(script_dir, 'Output');
cfg.results_dir = fullfile(cfg.output_dir, 'Results');
cfg.figure_root = fullfile(script_dir, 'Figures');
cfg.extended_fig_dir = fullfile(cfg.figure_root, 'Extended');

cfg.curve_csv = fullfile(cfg.results_dir, 'P4_Exp06_TrueSOH_ActionValueCurves.csv');
cfg.threshold_csv = fullfile(cfg.results_dir, 'P4_Exp06_TrueSOH_Thresholds.csv');
cfg.result_mat = fullfile(cfg.results_dir, 'P4_Exp06_TrueSOH_ActionValueBenchmark.mat');

cfg.fig_file_lfp = fullfile(cfg.extended_fig_dir, 'ExD04a_LFP_true_SOH_action_value_benchmark.fig');
cfg.png_file_lfp = fullfile(cfg.extended_fig_dir, 'ExD04a_LFP_true_SOH_action_value_benchmark.png');
cfg.fig_file_nmc = fullfile(cfg.extended_fig_dir, 'ExD04b_NMC_true_SOH_action_value_benchmark.fig');
cfg.png_file_nmc = fullfile(cfg.extended_fig_dir, 'ExD04b_NMC_true_SOH_action_value_benchmark.png');

cfg.png_resolution = 600;

folder_list = {cfg.output_dir, cfg.results_dir, cfg.figure_root, cfg.extended_fig_dir};

for ii = 1:numel(folder_list)
    if ~exist(folder_list{ii}, 'dir')
        mkdir(folder_list{ii});
    end
end

fprintf('================ P4 Exp06: true-SOH action-value benchmark ================\n');
fprintf('LFP SOH input: %s\n', cfg.lfp_soh_mat);
fprintf('NMC SOH input: %s\n', cfg.nmc_soh_mat);
fprintf('Output folder: %s\n', cfg.output_dir);
fprintf('Figure folder: %s\n', cfg.extended_fig_dir);

%% 2) Build true-SOH action-value curves for LFP and NMC
LFP = local_build_action_values('LFP', cfg.lfp_soh_mat);
NMC = local_build_action_values('NMC', cfg.nmc_soh_mat);

fprintf('\nLFP true-SOH thresholds:\n');
fprintf('T1,true = %.6f\n', LFP.T1_true);
fprintf('T2,true = %.6f\n', LFP.T2_true);

fprintf('\nNMC true-SOH thresholds:\n');
fprintf('T1,true = %.6f\n', NMC.T1_true);
fprintf('T2,true = %.6f\n', NMC.T2_true);

%% 3) Save curve and threshold tables
LFP_Curve_Table = local_make_curve_table(LFP);
NMC_Curve_Table = local_make_curve_table(NMC);

Curve_Table = [LFP_Curve_Table; NMC_Curve_Table];

Threshold_Table = table( ...
    ["LFP"; "NMC"], ...
    [LFP.T1_true; NMC.T1_true], ...
    [LFP.T2_true; NMC.T2_true], ...
    [LFP.params.beta_refurbish; NMC.params.beta_refurbish], ...
    [LFP.params.C_A1_extra; NMC.params.C_A1_extra], ...
    [LFP.params.C_A2_extra; NMC.params.C_A2_extra], ...
    [LFP.params.C0; NMC.params.C0], ...
    [LFP.params.C1; NMC.params.C1], ...
    [LFP.params.gamma_ref; NMC.params.gamma_ref], ...
    'VariableNames', { ...
    'Chemistry', ...
    'T1_true', ...
    'T2_true', ...
    'beta_refurbish', ...
    'C_A1_extra_USD_per_pack', ...
    'C_A2_extra_USD_per_pack', ...
    'C0_USD_per_pack', ...
    'C1_USD_per_pack', ...
    'gamma_ref'});

writetable(Curve_Table, cfg.curve_csv);
writetable(Threshold_Table, cfg.threshold_csv);

save(cfg.result_mat, ...
    'cfg', ...
    'LFP', ...
    'NMC', ...
    'Curve_Table', ...
    'Threshold_Table', ...
    '-v7.3');

fprintf('\nCurve table saved:     %s\n', cfg.curve_csv);
fprintf('Threshold table saved: %s\n', cfg.threshold_csv);
fprintf('MAT result saved:      %s\n', cfg.result_mat);

%% 4) Plot Extended Data Fig. 4a: LFP in a separate window
fig_lfp = figure( ...
    'Name', 'ExD04a_LFP_true_SOH_action_value_benchmark', ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [4, 4, 8.2, 6.8]);

ax_lfp = axes('Parent', fig_lfp);
local_plot_action_value_panel(ax_lfp, LFP, 'a', 'LFP');

local_save_figure(fig_lfp, ...
    cfg.fig_file_lfp, ...
    cfg.png_file_lfp, ...
    cfg.png_resolution);

fprintf('LFP figure saved:      %s\n', cfg.png_file_lfp);

%% 5) Plot Extended Data Fig. 4b: NMC in a separate window
fig_nmc = figure( ...
    'Name', 'ExD04b_NMC_true_SOH_action_value_benchmark', ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [4, 4, 8.2, 6.8]);

ax_nmc = axes('Parent', fig_nmc);
local_plot_action_value_panel(ax_nmc, NMC, 'b', 'NMC');

local_save_figure(fig_nmc, ...
    cfg.fig_file_nmc, ...
    cfg.png_file_nmc, ...
    cfg.png_resolution);

fprintf('NMC figure saved:      %s\n', cfg.png_file_nmc);
fprintf('=======================================================================\n');




%% Local helper functions
function D = local_build_action_values(chemistry, input_mat)
% Build true-SOH action-value curves and theoretical thresholds.

if ~exist(input_mat, 'file')
    error('%s SOH input file not found: %s', chemistry, input_mat);
end

S = load(input_mat);
SOH_data = local_extract_soh_from_mat(S, input_mat, chemistry);

params = local_define_parameters(chemistry);

[a, V1] = local_construct_reuse_curve(SOH_data, params);

calc_V_reuse_base = @(soh) interp1(a, V1, soh, 'linear', 'extrap');

calc_V_reuse = @(soh) calc_V_reuse_base(soh) ...
    - params.C_A1_extra;

calc_C_refurbish = @(soh) params.C0 ...
    + params.C1 .* (1 - soh).^params.gamma_ref;

calc_V_refurbish_base = @(soh) ...
    calc_V_reuse_base(min(soh + params.beta_refurbish .* (1 - soh), 1.0)) ...
    - calc_C_refurbish(soh);

calc_V_refurbish = @(soh) calc_V_refurbish_base(soh) ...
    - params.C_A1_extra ...
    - params.C_A2_extra;

if strcmpi(chemistry, 'LFP')

    Y_Li2CO3_func = @(soh) ...
        (params.PosAcMass / params.M_LiFePO4) .* ...
        (soh ./ 2) .* params.M_Li2CO3_half .* params.Rec_Rate_Chem;

    Y_H2SO4_func = @(soh) ...
        (params.PosAcMass / params.M_LiFePO4) .* ...
        (soh ./ 2) .* params.M_H2SO4;

    Y_H2O2_func = @(soh) ...
        (params.PosAcMass / params.M_LiFePO4) .* ...
        (soh ./ 2) .* params.M_H2O2;

    Y_Na2CO3_func = @(soh) ...
        (params.PosAcMass / params.M_LiFePO4) .* ...
        (soh ./ 2) .* params.M_Na2CO3;

    calc_V_recycle = @(soh) params.M_pack .* params.Exchange_Rate .* ( ...
        (Y_Li2CO3_func(soh) .* params.P_Li2CO3 + ...
        params.Base_Y_FePO4 .* params.P_FePO4 + ...
        params.Base_Y_Graphite .* params.P_Graphite + ...
        params.Base_Y_Al .* params.P_Al + ...
        params.Base_Y_Cu .* params.P_Cu) ...
        - (Y_H2SO4_func(soh) .* params.P_H2SO4 + ...
        Y_H2O2_func(soh) .* params.P_H2O2 + ...
        Y_Na2CO3_func(soh) .* params.P_Na2CO3) ...
        - (params.PosAcMass .* params.C_Degraded + params.Fixed_Op_Cost));

elseif strcmpi(chemistry, 'NMC')

    Y_Li2CO3_in_func = @(soh) ...
        (params.PosAcMass ./ params.M_NMC622) .* ...
        ((1 - soh) ./ 2) .* params.M_Li2CO3;

    calc_V_recycle = @(soh) params.M_pack .* params.Exchange_Rate .* ( ...
        (params.Base_Y_NMC622 .* params.P_NMC622 + ...
        params.Base_Y_Graphite .* params.P_Graphite + ...
        params.Base_Y_Al .* params.P_Al + ...
        params.Base_Y_Cu .* params.P_Cu) ...
        - (Y_Li2CO3_in_func(soh) .* params.P_Li2CO3) ...
        - (params.PosAcMass .* params.C_Degraded + params.Fixed_Op_Cost));

else
    error('Unknown chemistry: %s', chemistry);
end

soh_grid = (0.40:0.001:1.00)';

V_A1 = calc_V_reuse(soh_grid);
V_A2 = calc_V_refurbish(soh_grid);
V_A3 = calc_V_recycle(soh_grid);

V_A1 = V_A1(:);
V_A2 = V_A2(:);
V_A3 = V_A3(:);

valid_idx = isfinite(soh_grid) & ...
    isfinite(V_A1) & ...
    isfinite(V_A2) & ...
    isfinite(V_A3);

soh_grid = soh_grid(valid_idx);
V_A1 = V_A1(valid_idx);
V_A2 = V_A2(valid_idx);
V_A3 = V_A3(valid_idx);

smooth_span = 5;

if smooth_span > 1
    V_A1_plot = smoothdata(V_A1, 'movmean', smooth_span);
    V_A2_plot = smoothdata(V_A2, 'movmean', smooth_span);
    V_A3_plot = smoothdata(V_A3, 'movmean', smooth_span);
else
    V_A1_plot = V_A1;
    V_A2_plot = V_A2;
    V_A3_plot = V_A3;
end

[best_action, T1_true, T2_true] = local_find_true_thresholds( ...
    soh_grid, V_A1, V_A2, V_A3);

D = struct();
D.chemistry = chemistry;
D.input_mat = input_mat;
D.SOH_data = SOH_data;
D.params = params;
D.a = a;
D.V1 = V1;
D.soh_grid = soh_grid;
D.V_A1 = V_A1;
D.V_A2 = V_A2;
D.V_A3 = V_A3;
D.V_A1_plot = V_A1_plot;
D.V_A2_plot = V_A2_plot;
D.V_A3_plot = V_A3_plot;
D.best_action = best_action;
D.T1_true = T1_true;
D.T2_true = T2_true;
end

function params = local_define_parameters(chemistry)
% Define chemistry-specific physical and economic parameters.

params = struct();

params.chemistry = chemistry;

params.Q_pack = 25.3;
params.r = 0.05;
params.beta_refurbish = 0.09;

params.ExchangeRate_RMB_USD = 7.04;
params.p_high_RMB = 1.1621;
params.p_low_RMB = 0.2986;

params.p_high = params.p_high_RMB / params.ExchangeRate_RMB_USD;
params.p_low = params.p_low_RMB / params.ExchangeRate_RMB_USD;

params.M_pack = 173.25;
params.Exchange_Rate = 1;

params.P_Li2CO3 = 27.20;
params.P_Graphite = 3.84;
params.P_Al = 3.42;
params.P_Cu = 14.79;

if strcmpi(chemistry, 'LFP')

    params.C_A1_extra = 10;
    params.C_A2_extra = 15;

    params.C0 = 92;
    params.C1 = 65;
    params.gamma_ref = 3;

    params.efficiency_func = @(soh) -0.1532 .* (1 - soh).^2 + ...
        0.003923 .* (1 - soh) + 0.9530;

    params.PosAcMass = 0.223;
    params.PosCCMass = 0.060;
    params.NegAcMass = 0.1235;
    params.NegCCMass = 0.08;

    params.M_LiFePO4 = 157.76;
    params.M_Li2CO3_half = 73.88;
    params.M_FePO4 = 150.82;
    params.M_H2SO4 = 98;
    params.M_H2O2 = 34;
    params.M_Na2CO3 = 106;

    params.Rec_Rate_Chem = 0.98;
    params.Rec_Rate_Graphite = 0.95;
    params.Rec_Rate_Al = 0.93;
    params.Rec_Rate_Cu = 0.93;

    params.P_FePO4 = 1.91;

    params.Base_Y_FePO4 = ...
        (params.PosAcMass / params.M_LiFePO4) * ...
        params.M_FePO4 * params.Rec_Rate_Chem;

    params.Base_Y_Graphite = params.NegAcMass * params.Rec_Rate_Graphite;
    params.Base_Y_Al = params.PosCCMass * params.Rec_Rate_Al;
    params.Base_Y_Cu = params.NegCCMass * params.Rec_Rate_Cu;

    params.C_Degraded = 2.05;
    params.P_H2SO4 = 0.264;
    params.P_H2O2 = 0.130;
    params.P_Na2CO3 = 0.172;

    params.P_elec_recycle = 0.111;

    params.C_Electricity_hydro = 500 * params.P_elec_recycle / 1000;
    params.C_Labor_hydro = 307.36 / 1000;
    params.C_Disassemble_hydro = 12.07 / 1000;
    params.C_Equip_hydro = 52.00 / 1000;
    params.C_Sewage_hydro = 141.14 / 1000;

    params.Fixed_Op_Cost = params.C_Electricity_hydro + ...
        params.C_Labor_hydro + ...
        params.C_Disassemble_hydro + ...
        params.C_Equip_hydro + ...
        params.C_Sewage_hydro;

elseif strcmpi(chemistry, 'NMC')

    params.C_A1_extra = 10;
    params.C_A2_extra = 15;

    params.C0 = 43;
    params.C1 = 80;
    params.gamma_ref = 3;

    params.efficiency_func = @(soh) -0.2303 .* (1 - soh) + 0.9582;

    params.PosAcMass = 0.2314;
    params.PosCCMass = 0.0700;
    params.NegAcMass = 0.1425;
    params.NegCCMass = 0.1700;

    params.M_NMC622 = 96.93;
    params.M_Li2CO3 = 73.89;

    params.Rec_Rate_Cathode = 0.90;
    params.Rec_Rate_Graphite = 0.90;
    params.Rec_Rate_Al = 0.90;
    params.Rec_Rate_Cu = 0.90;

    params.P_NMC622 = 29.17;

    params.C_Degraded = 17.34;

    params.C_Electricity_direct = 300 * 0.111 / 1000;
    params.C_Labor_direct = 223.79 / 1000;
    params.C_Disassemble_direct = 143.00 / 1000;
    params.C_Equip_direct = 57.57 / 1000;
    params.C_Sewage_direct = 100.29 / 1000;

    params.Fixed_Op_Cost = params.C_Electricity_direct + ...
        params.C_Labor_direct + ...
        params.C_Disassemble_direct + ...
        params.C_Equip_direct + ...
        params.C_Sewage_direct;

    params.Base_Y_NMC622 = params.PosAcMass * params.Rec_Rate_Cathode;
    params.Base_Y_Graphite = params.NegAcMass * params.Rec_Rate_Graphite;
    params.Base_Y_Al = params.PosCCMass * params.Rec_Rate_Al;
    params.Base_Y_Cu = params.NegCCMass * params.Rec_Rate_Cu;

else
    error('Unknown chemistry: %s', chemistry);
end
end

function [a, V1] = local_construct_reuse_curve(SOH_data, params)
% Construct the reuse-value interpolation curve from an SOH trajectory.

valid_end_idx = find(SOH_data >= 0.4);

if isempty(valid_end_idx)
    error('SOH_data does not contain valid points with SOH >= 0.4.');
end

end_idx = valid_end_idx(end);

start_indices = find(SOH_data <= 1.0 & SOH_data >= 0.4);
num_cases = numel(start_indices);

if num_cases < 2
    error('SOH_data contains too few starting points for constructing the reuse-value curve.');
end

a = zeros(num_cases, 1);
V1 = zeros(num_cases, 1);

for kk = 1:num_cases

    start_idx = start_indices(kk);

    SOH_slice = SOH_data(start_idx:end_idx);
    SOH_slice = SOH_slice(:);

    a(kk) = SOH_slice(1);

    days_elapsed = (0:length(SOH_slice)-1)';

    eta_slice = params.efficiency_func(SOH_slice);

    daily_value_raw = params.Q_pack .* SOH_slice .* ...
        (eta_slice .* params.p_high - params.p_low);

    discount_factor = 1 ./ ((1 + params.r) .^ (days_elapsed / 365));

    V1(kk) = sum(daily_value_raw .* discount_factor);
end

[a, sort_idx] = sort(a);
V1 = V1(sort_idx);

[a_unique, ia] = unique(a, 'last');
V1_unique = V1(ia);

a = a_unique;
V1 = V1_unique;

if numel(a) < 2
    error('Too few SOH starting points remain after duplicate removal.');
end

if max(a) < 1.0

    slope_end = (V1(end) - V1(end - 1)) / ...
        (a(end) - a(end - 1));

    V1_at_1 = V1(end) + slope_end * (1.0 - a(end));

    a = [a; 1.0];
    V1 = [V1; V1_at_1];
end
end

function [best_action, T1_true, T2_true] = local_find_true_thresholds( ...
    soh_grid, V_A1, V_A2, V_A3)
% Find theoretical action-switching thresholds on the upper envelope.

V_all = [V_A1, V_A2, V_A3];

[~, best_action] = max(V_all, [], 2);

T1_true = NaN;
T2_true = NaN;

idx_T1_env = find(best_action(1:end-1) == 3 & ...
    best_action(2:end) == 2, 1, 'first');

if ~isempty(idx_T1_env)

    idx_cross_T1 = idx_T1_env;

else

    diff_A2_A3 = V_A2 - V_A3;
    idx_cross_T1_all = find(diff_A2_A3(1:end-1) .* ...
        diff_A2_A3(2:end) <= 0);

    if ~isempty(idx_cross_T1_all)
        idx_cross_T1 = idx_cross_T1_all(1);
    else
        idx_cross_T1 = [];
    end
end

if ~isempty(idx_cross_T1)

    x_pair = soh_grid(idx_cross_T1:idx_cross_T1 + 1);
    y_pair = V_A2(idx_cross_T1:idx_cross_T1 + 1) - ...
        V_A3(idx_cross_T1:idx_cross_T1 + 1);

    if abs(y_pair(2) - y_pair(1)) > eps
        T1_true = interp1(y_pair, x_pair, 0, 'linear', 'extrap');
    else
        T1_true = mean(x_pair);
    end
end

idx_T2_env = find(best_action(1:end-1) == 2 & ...
    best_action(2:end) == 1, 1, 'first');

if ~isempty(idx_T2_env)

    idx_cross_T2 = idx_T2_env;

else

    diff_A1_A2 = V_A1 - V_A2;
    idx_cross_T2_all = find(diff_A1_A2(1:end-1) .* ...
        diff_A1_A2(2:end) <= 0);

    if ~isempty(idx_cross_T2_all)

        if isfinite(T1_true)

            idx_after_T1 = idx_cross_T2_all( ...
                soh_grid(idx_cross_T2_all) > T1_true);

            if ~isempty(idx_after_T1)
                idx_cross_T2 = idx_after_T1(1);
            else
                idx_cross_T2 = idx_cross_T2_all(1);
            end

        else

            idx_cross_T2 = idx_cross_T2_all(1);
        end

    else

        idx_cross_T2 = [];
    end
end

if ~isempty(idx_cross_T2)

    x_pair = soh_grid(idx_cross_T2:idx_cross_T2 + 1);
    y_pair = V_A1(idx_cross_T2:idx_cross_T2 + 1) - ...
        V_A2(idx_cross_T2:idx_cross_T2 + 1);

    if abs(y_pair(2) - y_pair(1)) > eps
        T2_true = interp1(y_pair, x_pair, 0, 'linear', 'extrap');
    else
        T2_true = mean(x_pair);
    end
end
end

function Curve_Table = local_make_curve_table(D)
% Convert action-value curves to a table.

n = numel(D.soh_grid);

Curve_Table = table( ...
    repmat(string(D.chemistry), n, 1), ...
    D.soh_grid, ...
    D.V_A1, ...
    D.V_A2, ...
    D.V_A3, ...
    D.V_A1_plot, ...
    D.V_A2_plot, ...
    D.V_A3_plot, ...
    D.best_action, ...
    'VariableNames', { ...
    'Chemistry', ...
    'True_SOH', ...
    'A1_direct_reuse_value_USD_per_pack', ...
    'A2_refurbishment_value_USD_per_pack', ...
    'A3_recycling_value_USD_per_pack', ...
    'A1_direct_reuse_value_smoothed_USD_per_pack', ...
    'A2_refurbishment_value_smoothed_USD_per_pack', ...
    'A3_recycling_value_smoothed_USD_per_pack', ...
    'Envelope_best_action'});
end

function local_plot_action_value_panel(ax, D, panel_label, panel_title)
% Plot one chemistry-specific action-value panel.

axes(ax);
hold(ax, 'on');
box(ax, 'on');
grid(ax, 'on');

color_A1 = [0.00, 0.45, 0.74];
color_A2 = [0.47, 0.67, 0.19];
color_A3 = [0.85, 0.33, 0.10];

p1 = plot(ax, D.soh_grid, D.V_A1_plot, '-', ...
    'LineWidth', 2.0, ...
    'Color', color_A1);

p2 = plot(ax, D.soh_grid, D.V_A2_plot, '-', ...
    'LineWidth', 2.0, ...
    'Color', color_A2);

p3 = plot(ax, D.soh_grid, D.V_A3_plot, '-', ...
    'LineWidth', 2.0, ...
    'Color', color_A3);

all_values = [D.V_A1_plot; D.V_A2_plot; D.V_A3_plot];

y_min = min(all_values);
y_max = max(all_values);
y_range = y_max - y_min;

if y_range <= 0 || ~isfinite(y_range)
    y_range = 1;
end

if y_min > 0
    y_lower = 0;
else
    y_lower = y_min - 0.08 * y_range;
end

y_upper = y_max + 0.10 * y_range;

xlim(ax, [0.40, 1.00]);
ylim(ax, [y_lower, y_upper]);

if isfinite(D.T1_true)

    xline(ax, D.T1_true, '--', ...
        'Color', [0.20, 0.35, 0.95], ...
        'LineWidth', 1.2, ...
        'HandleVisibility', 'off');

    text(ax, D.T1_true + 0.008, ...
        y_lower + 0.16 * (y_upper - y_lower), ...
        sprintf('T_{1,true} = %.3f', D.T1_true), ...
        'FontName', 'Arial', ...
        'FontSize', 8.5, ...
        'Color', [0.20, 0.35, 0.95]);
end

if isfinite(D.T2_true)

    xline(ax, D.T2_true, '--', ...
        'Color', [0.95, 0.25, 0.25], ...
        'LineWidth', 1.2, ...
        'HandleVisibility', 'off');

    text(ax, D.T2_true - 0.150, ...
        y_lower + 0.82 * (y_upper - y_lower), ...
        sprintf('T_{2,true} = %.3f', D.T2_true), ...
        'FontName', 'Arial', ...
        'FontSize', 8.5, ...
        'Color', [0.95, 0.25, 0.25]);
end

xlabel(ax, 'True SOH', ...
    'FontName', 'Arial', ...
    'FontSize', 9);

ylabel(ax, 'Net value (USD per pack)', ...
    'FontName', 'Arial', ...
    'FontSize', 9);

title(ax, panel_title, ...
    'FontName', 'Arial', ...
    'FontSize', 10, ...
    'FontWeight', 'normal');

legend(ax, [p1, p2, p3], ...
    {'A_1: Direct reuse', 'A_2: Refurbishment', 'A_3: Recycling'}, ...
    'Location', 'northwest', ...
    'Box', 'off', ...
    'FontName', 'Arial', ...
    'FontSize', 7.5);

set(ax, ...
    'FontName', 'Arial', ...
    'FontSize', 8.5, ...
    'LineWidth', 0.9, ...
    'TickDir', 'out');
end

function SOH_data = local_extract_soh_from_mat(SOH_file, input_file, chemistry)
% Extract a numeric SOH vector from a MAT-file structure.

if strcmpi(chemistry, 'LFP')

    if isfield(SOH_file, 'SOH_array')

        SOH_raw = SOH_file.SOH_array;

    elseif isfield(SOH_file, 'LFP_SOH_Table')

        SOH_raw = SOH_file.LFP_SOH_Table;

    elseif isfield(SOH_file, 'SOH_data')

        SOH_raw = SOH_file.SOH_data;

    else

        var_names = fieldnames(SOH_file);

        if isempty(var_names)
            error('No variables were found in %s.', input_file);
        end

        SOH_raw = SOH_file.(var_names{1});
    end

elseif strcmpi(chemistry, 'NMC')

    if isfield(SOH_file, 'SOH_table')

        SOH_raw = SOH_file.SOH_table;

        if isfield(SOH_file, 'day')
            day_count = SOH_file.day;
            SOH_raw = SOH_raw(1:day_count);
        end

    elseif isfield(SOH_file, 'NMC_SOH_Table')

        SOH_raw = SOH_file.NMC_SOH_Table;

    elseif isfield(SOH_file, 'SOH_data')

        SOH_raw = SOH_file.SOH_data;

    else

        var_names = fieldnames(SOH_file);

        if isempty(var_names)
            error('No variables were found in %s.', input_file);
        end

        SOH_raw = SOH_file.(var_names{1});
    end

else
    error('Unknown chemistry: %s', chemistry);
end

if istable(SOH_raw)

    table_vars = SOH_raw.Properties.VariableNames;
    soh_col_idx = find(contains(lower(table_vars), 'soh'), 1, 'first');

    if isempty(soh_col_idx)
        soh_col_idx = 1;
    end

    SOH_data = SOH_raw{:, soh_col_idx};

else

    SOH_data = SOH_raw;
end

SOH_data = double(SOH_data(:));
SOH_data = SOH_data(isfinite(SOH_data));

if isempty(SOH_data)
    error('No valid %s SOH values were found in %s.', chemistry, input_file);
end
end

function local_save_figure(fig_handle, fig_file, png_file, resolution)
% Save a figure as both FIG and PNG files.

fig_dir = fileparts(fig_file);

if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

savefig(fig_handle, fig_file);
exportgraphics(fig_handle, png_file, 'Resolution', resolution);

fprintf('[Figure] Saved: %s\n', png_file);
end