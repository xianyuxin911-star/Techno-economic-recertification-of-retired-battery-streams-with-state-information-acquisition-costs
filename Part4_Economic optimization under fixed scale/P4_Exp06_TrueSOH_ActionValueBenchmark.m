%% =========================================================================
% P4_Exp06_TrueSOH_ActionValueBenchmark.m
%
% Purpose:
% This script constructs the perfect-information, true-SOH action-value
% benchmark for LFP and NMC recertification pathways and generates
% Extended Data Fig. 4.
%
% The benchmark uses the same chemistry-specific economic assumptions as
% Part4 Exp01 and Exp02, but removes prediction error and information-
% acquisition cost. At each true SOH, the preferred action is the pathway
% with the largest net action value.
%
% Outputs:
%   1) Chemistry-specific action-value curves;
%   2) True-SOH pathway-switching thresholds;
%   3) Threshold diagnostics and crossing validation;
%   4) Extended Data Fig. 4a-b.
%% =========================================================================

clear;
clc;
close all;

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

cfg.curve_csv = fullfile(cfg.results_dir, ...
    'P4_Exp06_TrueSOH_ActionValueCurves.csv');

cfg.threshold_csv = fullfile(cfg.results_dir, ...
    'P4_Exp06_TrueSOH_Thresholds.csv');

cfg.diagnostic_csv = fullfile(cfg.results_dir, ...
    'P4_Exp06_TrueSOH_ThresholdDiagnostics.csv');

cfg.result_mat = fullfile(cfg.results_dir, ...
    'P4_Exp06_TrueSOH_ActionValueBenchmark_Results.mat');

cfg.fig_file_lfp = fullfile(cfg.extended_fig_dir, ...
    'ExD04a_LFP_true_SOH_action_value_benchmark.fig');

cfg.png_file_lfp = fullfile(cfg.extended_fig_dir, ...
    'ExD04a_LFP_true_SOH_action_value_benchmark.png');

cfg.fig_file_nmc = fullfile(cfg.extended_fig_dir, ...
    'ExD04b_NMC_true_SOH_action_value_benchmark.fig');

cfg.png_file_nmc = fullfile(cfg.extended_fig_dir, ...
    'ExD04b_NMC_true_SOH_action_value_benchmark.png');

% True-SOH evaluation grid.
cfg.soh_min = 0.40;
cfg.soh_max = 1.00;
cfg.soh_step = 0.001;

% Deterministic value functions are already smooth. Keep display smoothing
% disabled by default so the plotted curves and calculated thresholds use
% the same underlying values.
cfg.plot_smoothing_span = 1;

cfg.png_resolution = 600;

folder_list = { ...
    cfg.output_dir, ...
    cfg.results_dir, ...
    cfg.figure_root, ...
    cfg.extended_fig_dir};

for ii = 1:numel(folder_list)

    if ~exist(folder_list{ii}, 'dir')

        [status, message] = mkdir(folder_list{ii});

        if ~status
            error('Unable to create folder:\n%s\nReason: %s', ...
                folder_list{ii}, message);
        end
    end
end

fprintf('================ P4 Exp06: true-SOH action-value benchmark ================\n');
fprintf('LFP SOH input:        %s\n', cfg.lfp_soh_mat);
fprintf('NMC SOH input:        %s\n', cfg.nmc_soh_mat);
fprintf('SOH evaluation grid:  %.3f to %.3f, step %.4f\n', ...
    cfg.soh_min, cfg.soh_max, cfg.soh_step);
fprintf('Results folder:       %s\n', cfg.results_dir);
fprintf('Figure folder:        %s\n', cfg.extended_fig_dir);

%% 2) Build true-SOH action-value benchmarks
LFP = local_build_action_values('LFP', cfg.lfp_soh_mat, cfg);
NMC = local_build_action_values('NMC', cfg.nmc_soh_mat, cfg);

fprintf('\n================ True-SOH thresholds ================\n');
fprintf('LFP: T1,true = %.6f | T2,true = %.6f\n', ...
    LFP.T1_true, LFP.T2_true);
fprintf('NMC: T1,true = %.6f | T2,true = %.6f\n', ...
    NMC.T1_true, NMC.T2_true);
fprintf('=====================================================\n');

%% 3) Build and save output tables
LFP_Curve_Table = local_make_curve_table(LFP);
NMC_Curve_Table = local_make_curve_table(NMC);

Curve_Table = [LFP_Curve_Table; NMC_Curve_Table];

Threshold_Table = table( ...
    ["LFP"; "NMC"], ...
    [LFP.T1_true; NMC.T1_true], ...
    [LFP.T2_true; NMC.T2_true], ...
    [LFP.threshold_diagnostics.T1_transition_detected; ...
     NMC.threshold_diagnostics.T1_transition_detected], ...
    [LFP.threshold_diagnostics.T2_transition_detected; ...
     NMC.threshold_diagnostics.T2_transition_detected], ...
    [LFP.threshold_diagnostics.T1_selected_from_pairwise_crossing; ...
     NMC.threshold_diagnostics.T1_selected_from_pairwise_crossing], ...
    [LFP.threshold_diagnostics.T2_selected_from_pairwise_crossing; ...
     NMC.threshold_diagnostics.T2_selected_from_pairwise_crossing], ...
    [string(LFP.threshold_diagnostics.envelope_action_sequence); ...
     string(NMC.threshold_diagnostics.envelope_action_sequence)], ...
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
    'T1_envelope_transition_detected', ...
    'T2_envelope_transition_detected', ...
    'T1_selected_from_A2_A3_crossing', ...
    'T2_selected_from_A1_A2_crossing', ...
    'Envelope_action_sequence', ...
    'beta_refurbish', ...
    'C_A1_extra_USD_per_pack', ...
    'C_A2_extra_USD_per_pack', ...
    'C0_USD_per_pack', ...
    'C1_USD_per_pack', ...
    'gamma_ref'});

Threshold_Diagnostic_Table = local_make_threshold_diagnostic_table( ...
    LFP, NMC);

writetable(Curve_Table, cfg.curve_csv);
writetable(Threshold_Table, cfg.threshold_csv);
writetable(Threshold_Diagnostic_Table, cfg.diagnostic_csv);

save(cfg.result_mat, ...
    'cfg', ...
    'LFP', ...
    'NMC', ...
    'Curve_Table', ...
    'Threshold_Table', ...
    'Threshold_Diagnostic_Table', ...
    '-v7.3');

fprintf('\nCurve table saved:       %s\n', cfg.curve_csv);
fprintf('Threshold table saved:   %s\n', cfg.threshold_csv);
fprintf('Diagnostic table saved:  %s\n', cfg.diagnostic_csv);
fprintf('MAT result saved:        %s\n', cfg.result_mat);

%% 4) Plot Extended Data Fig. 4a: LFP
fig_lfp = figure( ...
    'Name', 'ExD04a_LFP_true_SOH_action_value_benchmark', ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [4, 4, 8.2, 6.8]);

ax_lfp = axes('Parent', fig_lfp);

local_plot_action_value_panel( ...
    ax_lfp, LFP, 'a', 'LFP');

local_save_figure( ...
    fig_lfp, ...
    cfg.fig_file_lfp, ...
    cfg.png_file_lfp, ...
    cfg.png_resolution);

fprintf('LFP figure saved:        %s\n', cfg.png_file_lfp);

%% 5) Plot Extended Data Fig. 4b: NMC
fig_nmc = figure( ...
    'Name', 'ExD04b_NMC_true_SOH_action_value_benchmark', ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [4, 4, 8.2, 6.8]);

ax_nmc = axes('Parent', fig_nmc);

local_plot_action_value_panel( ...
    ax_nmc, NMC, 'b', 'NMC');

local_save_figure( ...
    fig_nmc, ...
    cfg.fig_file_nmc, ...
    cfg.png_file_nmc, ...
    cfg.png_resolution);

fprintf('NMC figure saved:        %s\n', cfg.png_file_nmc);
fprintf('================ P4 Exp06 completed ================\n');

%% =========================================================================
% Local helper functions
%% =========================================================================

function D = local_build_action_values(chemistry, input_mat, cfg)
% Build chemistry-specific true-SOH action-value curves and thresholds.

if ~exist(input_mat, 'file')
    error('%s SOH input file not found: %s', chemistry, input_mat);
end

S = load(input_mat);
SOH_data = local_extract_soh_from_mat(S, input_mat, chemistry);

params = local_define_parameters(chemistry);

[a, V1] = local_construct_reuse_curve(SOH_data, params);

calc_V_reuse_base = @(soh) interp1( ...
    a, V1, soh, 'linear', 'extrap');

calc_V_reuse = @(soh) ...
    calc_V_reuse_base(soh) ...
    - params.C_A1_extra;

calc_C_refurbish = @(soh) ...
    params.C0 ...
    + params.C1 .* (1 - soh).^params.gamma_ref;

calc_V_refurbish_base = @(soh) ...
    calc_V_reuse_base( ...
    min(soh + params.beta_refurbish .* (1 - soh), 1.0)) ...
    - calc_C_refurbish(soh);

calc_V_refurbish = @(soh) ...
    calc_V_refurbish_base(soh) ...
    - params.C_A1_extra ...
    - params.C_A2_extra;

if strcmpi(chemistry, 'LFP')

    Y_Li2CO3_func = @(soh) ...
        (params.PosAcMass / params.M_LiFePO4) .* ...
        (soh ./ 2) .* ...
        params.M_Li2CO3_half .* ...
        params.Rec_Rate_Chem;

    Y_H2SO4_func = @(soh) ...
        (params.PosAcMass / params.M_LiFePO4) .* ...
        (soh ./ 2) .* ...
        params.M_H2SO4;

    Y_H2O2_func = @(soh) ...
        (params.PosAcMass / params.M_LiFePO4) .* ...
        (soh ./ 2) .* ...
        params.M_H2O2;

    Y_Na2CO3_func = @(soh) ...
        (params.PosAcMass / params.M_LiFePO4) .* ...
        (soh ./ 2) .* ...
        params.M_Na2CO3;

    calc_V_recycle = @(soh) ...
        params.M_pack .* params.Exchange_Rate .* ( ...
        ( ...
        Y_Li2CO3_func(soh) .* params.P_Li2CO3 + ...
        params.Base_Y_FePO4 .* params.P_FePO4 + ...
        params.Base_Y_Graphite .* params.P_Graphite + ...
        params.Base_Y_Al .* params.P_Al + ...
        params.Base_Y_Cu .* params.P_Cu ...
        ) ...
        - ( ...
        Y_H2SO4_func(soh) .* params.P_H2SO4 + ...
        Y_H2O2_func(soh) .* params.P_H2O2 + ...
        Y_Na2CO3_func(soh) .* params.P_Na2CO3 ...
        ) ...
        - ( ...
        params.PosAcMass .* params.C_Degraded + ...
        params.Fixed_Op_Cost ...
        ));

elseif strcmpi(chemistry, 'NMC')

    Y_Li2CO3_in_func = @(soh) ...
        (params.PosAcMass ./ params.M_NMC622) .* ...
        ((1 - soh) ./ 2) .* ...
        params.M_Li2CO3;

    calc_V_recycle = @(soh) ...
        params.M_pack .* params.Exchange_Rate .* ( ...
        ( ...
        params.Base_Y_NMC622 .* params.P_NMC622 + ...
        params.Base_Y_Graphite .* params.P_Graphite + ...
        params.Base_Y_Al .* params.P_Al + ...
        params.Base_Y_Cu .* params.P_Cu ...
        ) ...
        - (Y_Li2CO3_in_func(soh) .* params.P_Li2CO3) ...
        - ( ...
        params.PosAcMass .* params.C_Degraded + ...
        params.Fixed_Op_Cost ...
        ));

else
    error('Unknown chemistry: %s', chemistry);
end

soh_grid = (cfg.soh_min:cfg.soh_step:cfg.soh_max)';

V_A1 = calc_V_reuse(soh_grid);
V_A2 = calc_V_refurbish(soh_grid);
V_A3 = calc_V_recycle(soh_grid);

V_A1 = double(V_A1(:));
V_A2 = double(V_A2(:));
V_A3 = double(V_A3(:));

valid_idx = isfinite(soh_grid) & ...
    isfinite(V_A1) & ...
    isfinite(V_A2) & ...
    isfinite(V_A3);

soh_grid = soh_grid(valid_idx);
V_A1 = V_A1(valid_idx);
V_A2 = V_A2(valid_idx);
V_A3 = V_A3(valid_idx);

if numel(soh_grid) < 3
    error('Too few finite action-value points were generated for %s.', ...
        chemistry);
end

if cfg.plot_smoothing_span > 1

    V_A1_plot = smoothdata( ...
        V_A1, 'movmean', cfg.plot_smoothing_span);

    V_A2_plot = smoothdata( ...
        V_A2, 'movmean', cfg.plot_smoothing_span);

    V_A3_plot = smoothdata( ...
        V_A3, 'movmean', cfg.plot_smoothing_span);

else

    V_A1_plot = V_A1;
    V_A2_plot = V_A2;
    V_A3_plot = V_A3;
end

[best_action, T1_true, T2_true, threshold_diagnostics] = ...
    local_find_true_thresholds( ...
    soh_grid, ...
    V_A1, ...
    V_A2, ...
    V_A3);

if ~isfinite(T1_true) || ~isfinite(T2_true)
    warning('%s true-SOH threshold extraction returned a non-finite value.', ...
        chemistry);
elseif T1_true >= T2_true
    error(['Invalid %s true-SOH thresholds: T1,true = %.6f and ' ...
        'T2,true = %.6f. Expected T1,true < T2,true.'], ...
        chemistry, T1_true, T2_true);
end

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

D.gap_A2_minus_A3 = V_A2 - V_A3;
D.gap_A1_minus_A2 = V_A1 - V_A2;

D.best_action = best_action;

D.T1_true = T1_true;
D.T2_true = T2_true;

D.threshold_diagnostics = threshold_diagnostics;
end

function params = local_define_parameters(chemistry)
% Define chemistry-specific physical and economic parameters.
%
% These values must remain synchronized with Part4 Exp01 and Exp02.

params = struct();

params.chemistry = chemistry;

params.Q_pack = 25.3;
params.r = 0.05;
params.beta_refurbish = 0.09;

params.ExchangeRate_RMB_USD = 7.04;
params.p_high_RMB = 1.1621;
params.p_low_RMB = 0.2986;

params.p_high = ...
    params.p_high_RMB / params.ExchangeRate_RMB_USD;

params.p_low = ...
    params.p_low_RMB / params.ExchangeRate_RMB_USD;

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

    params.efficiency_func = @(soh) ...
        -0.1532 .* (1 - soh).^2 + ...
        0.003923 .* (1 - soh) + ...
        0.9530;

    params.PosAcMass = 0.223;
    params.PosCCMass = 0.060;
    params.NegAcMass = 0.1235;
    params.NegCCMass = 0.080;

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
        (params.PosAcMass / params.M_LiFePO4) .* ...
        params.M_FePO4 .* ...
        params.Rec_Rate_Chem;

    params.Base_Y_Graphite = ...
        params.NegAcMass .* params.Rec_Rate_Graphite;

    params.Base_Y_Al = ...
        params.PosCCMass .* params.Rec_Rate_Al;

    params.Base_Y_Cu = ...
        params.NegCCMass .* params.Rec_Rate_Cu;

    params.C_Degraded = 2.05;

    params.P_H2SO4 = 0.264;
    params.P_H2O2 = 0.130;
    params.P_Na2CO3 = 0.172;

    params.P_elec_recycle = 0.111;

    params.C_Electricity_hydro = ...
        500 .* params.P_elec_recycle ./ 1000;

    params.C_Labor_hydro = 307.36 ./ 1000;
    params.C_Disassemble_hydro = 12.07 ./ 1000;
    params.C_Equip_hydro = 52.00 ./ 1000;
    params.C_Sewage_hydro = 141.14 ./ 1000;

    params.Fixed_Op_Cost = ...
        params.C_Electricity_hydro + ...
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

    params.efficiency_func = @(soh) ...
        -0.2303 .* (1 - soh) + ...
        0.9582;

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

    params.C_Electricity_direct = 300 .* 0.111 ./ 1000;
    params.C_Labor_direct = 223.79 ./ 1000;
    params.C_Disassemble_direct = 143.00 ./ 1000;
    params.C_Equip_direct = 57.57 ./ 1000;
    params.C_Sewage_direct = 100.29 ./ 1000;

    params.Fixed_Op_Cost = ...
        params.C_Electricity_direct + ...
        params.C_Labor_direct + ...
        params.C_Disassemble_direct + ...
        params.C_Equip_direct + ...
        params.C_Sewage_direct;

    params.Base_Y_NMC622 = ...
        params.PosAcMass .* params.Rec_Rate_Cathode;

    params.Base_Y_Graphite = ...
        params.NegAcMass .* params.Rec_Rate_Graphite;

    params.Base_Y_Al = ...
        params.PosCCMass .* params.Rec_Rate_Al;

    params.Base_Y_Cu = ...
        params.NegCCMass .* params.Rec_Rate_Cu;

else
    error('Unknown chemistry: %s', chemistry);
end
end

function [a, V1] = local_construct_reuse_curve(SOH_data, params)
% Construct the reuse-value interpolation curve from an SOH trajectory.

valid_end_idx = find(SOH_data >= 0.4);

if isempty(valid_end_idx)
    error('SOH_data does not contain any points with SOH >= 0.4.');
end

end_idx = valid_end_idx(end);

start_indices = find( ...
    SOH_data <= 1.0 & ...
    SOH_data >= 0.4);

num_cases = numel(start_indices);

if num_cases < 2
    error(['SOH_data contains too few valid starting points for ' ...
        'constructing the reuse-value curve.']);
end

a = zeros(num_cases, 1);
V1 = zeros(num_cases, 1);

for kk = 1:num_cases

    start_idx = start_indices(kk);

    SOH_slice = SOH_data(start_idx:end_idx);
    SOH_slice = SOH_slice(:);

    if isempty(SOH_slice)
        error('An empty SOH trajectory slice was encountered.');
    end

    a(kk) = SOH_slice(1);

    days_elapsed = (0:numel(SOH_slice) - 1)';

    eta_slice = params.efficiency_func(SOH_slice);

    daily_value_raw = ...
        params.Q_pack .* ...
        SOH_slice .* ...
        (eta_slice .* params.p_high - params.p_low);

    discount_factor = ...
        1 ./ ((1 + params.r) .^ (days_elapsed ./ 365));

    V1(kk) = sum( ...
        daily_value_raw .* discount_factor);
end

[a, sort_idx] = sort(a);
V1 = V1(sort_idx);

[a, unique_idx] = unique(a, 'last');
V1 = V1(unique_idx);

if numel(a) < 2
    error('Too few SOH starting points remain after duplicate removal.');
end

if max(a) < 1.0

    denominator = a(end) - a(end - 1);

    if abs(denominator) <= eps
        error('Unable to extrapolate the reuse curve to SOH = 1.');
    end

    slope_end = ...
        (V1(end) - V1(end - 1)) ./ denominator;

    V1_at_1 = ...
        V1(end) + slope_end .* (1.0 - a(end));

    a = [a; 1.0];
    V1 = [V1; V1_at_1];
end
end

function [best_action, T1_true, T2_true, diagnostics] = ...
    local_find_true_thresholds(soh_grid, V_A1, V_A2, V_A3)
% Find true-SOH pathway-switching thresholds on the upper envelope.
%
% T1,true is the A3-to-A2 transition.
% T2,true is the A2-to-A1 transition.

V_all = [V_A1, V_A2, V_A3];

[~, best_action] = max(V_all, [], 2);

compressed_sequence = best_action([true; diff(best_action) ~= 0]);
sequence_string = strjoin( ...
    compose('A%d', compressed_sequence), ' -> ');

gap_A2_A3 = V_A2 - V_A3;
gap_A1_A2 = V_A1 - V_A2;

crossings_A2_A3 = local_find_zero_crossings( ...
    soh_grid, gap_A2_A3);

crossings_A1_A2 = local_find_zero_crossings( ...
    soh_grid, gap_A1_A2);

% Locate upper-envelope transitions.
idx_T1_transition = find( ...
    best_action(1:end - 1) == 3 & ...
    best_action(2:end) == 2, ...
    1, 'first');

idx_T2_transition = find( ...
    best_action(1:end - 1) == 2 & ...
    best_action(2:end) == 1, ...
    1, 'first');

T1_transition_detected = ~isempty(idx_T1_transition);
T2_transition_detected = ~isempty(idx_T2_transition);

T1_true = NaN;
T2_true = NaN;

T1_selected_from_pairwise_crossing = false;
T2_selected_from_pairwise_crossing = false;

% Select the A2-A3 crossing nearest the observed A3-to-A2 envelope switch.
if T1_transition_detected

    transition_midpoint_T1 = mean( ...
        soh_grid(idx_T1_transition:idx_T1_transition + 1));

    if ~isempty(crossings_A2_A3)

        [~, nearest_idx] = min( ...
            abs(crossings_A2_A3 - transition_midpoint_T1));

        T1_true = crossings_A2_A3(nearest_idx);
        T1_selected_from_pairwise_crossing = true;

    else

        T1_true = transition_midpoint_T1;
    end

elseif ~isempty(crossings_A2_A3)

    T1_true = crossings_A2_A3(1);
    T1_selected_from_pairwise_crossing = true;
end

% Select the A1-A2 crossing nearest the observed A2-to-A1 envelope switch.
if T2_transition_detected

    transition_midpoint_T2 = mean( ...
        soh_grid(idx_T2_transition:idx_T2_transition + 1));

    if ~isempty(crossings_A1_A2)

        [~, nearest_idx] = min( ...
            abs(crossings_A1_A2 - transition_midpoint_T2));

        T2_true = crossings_A1_A2(nearest_idx);
        T2_selected_from_pairwise_crossing = true;

    else

        T2_true = transition_midpoint_T2;
    end

elseif ~isempty(crossings_A1_A2)

    if isfinite(T1_true)

        candidate_idx = find( ...
            crossings_A1_A2 > T1_true, ...
            1, 'first');

        if isempty(candidate_idx)
            candidate_idx = 1;
        end

    else

        candidate_idx = 1;
    end

    T2_true = crossings_A1_A2(candidate_idx);
    T2_selected_from_pairwise_crossing = true;
end

diagnostics = struct();

diagnostics.envelope_action_sequence = sequence_string;

diagnostics.T1_transition_detected = ...
    T1_transition_detected;

diagnostics.T2_transition_detected = ...
    T2_transition_detected;

diagnostics.T1_selected_from_pairwise_crossing = ...
    T1_selected_from_pairwise_crossing;

diagnostics.T2_selected_from_pairwise_crossing = ...
    T2_selected_from_pairwise_crossing;

diagnostics.crossings_A2_A3 = ...
    crossings_A2_A3;

diagnostics.crossings_A1_A2 = ...
    crossings_A1_A2;

diagnostics.number_of_A2_A3_crossings = ...
    numel(crossings_A2_A3);

diagnostics.number_of_A1_A2_crossings = ...
    numel(crossings_A1_A2);

diagnostics.expected_monotonic_sequence = ...
    isequal(compressed_sequence(:)', [3, 2, 1]);

if ~diagnostics.expected_monotonic_sequence

    warning(['The upper-envelope action sequence is "%s" rather than ' ...
        '"A3 -> A2 -> A1". Inspect the action-value assumptions before ' ...
        'using the two-threshold interpretation.'], ...
        sequence_string);
end
end

function crossings = local_find_zero_crossings(x, y)
% Find all linearly interpolated zero crossings in y(x).

x = double(x(:));
y = double(y(:));

crossings = [];

for ii = 1:(numel(x) - 1)

    x1 = x(ii);
    x2 = x(ii + 1);

    y1 = y(ii);
    y2 = y(ii + 1);

    if ~isfinite(x1) || ~isfinite(x2) || ...
            ~isfinite(y1) || ~isfinite(y2)
        continue;
    end

    if y1 == 0

        crossings(end + 1, 1) = x1; %#ok<AGROW>

    elseif y1 .* y2 < 0

        crossing_now = ...
            x1 - y1 .* (x2 - x1) ./ (y2 - y1);

        crossings(end + 1, 1) = crossing_now; %#ok<AGROW>
    end
end

if isfinite(y(end)) && y(end) == 0
    crossings(end + 1, 1) = x(end); %#ok<AGROW>
end

if ~isempty(crossings)
    crossings = unique(round(crossings, 8), 'stable');
end
end

function Curve_Table = local_make_curve_table(D)
% Convert chemistry-specific action-value curves to a table.

n = numel(D.soh_grid);

Curve_Table = table( ...
    repmat(string(D.chemistry), n, 1), ...
    D.soh_grid, ...
    D.V_A1, ...
    D.V_A2, ...
    D.V_A3, ...
    D.gap_A2_minus_A3, ...
    D.gap_A1_minus_A2, ...
    D.best_action, ...
    'VariableNames', { ...
    'Chemistry', ...
    'True_SOH', ...
    'A1_direct_reuse_value_USD_per_pack', ...
    'A2_refurbishment_value_USD_per_pack', ...
    'A3_recycling_value_USD_per_pack', ...
    'A2_minus_A3_value_gap_USD_per_pack', ...
    'A1_minus_A2_value_gap_USD_per_pack', ...
    'Perfect_information_preferred_action'});
end

function Diagnostic_Table = local_make_threshold_diagnostic_table(LFP, NMC)
% Create a compact crossing-diagnostic table.

chemistry_list = ["LFP"; "NMC"];
data_list = {LFP; NMC};

T1_true = nan(2, 1);
T2_true = nan(2, 1);

Number_A2_A3_crossings = zeros(2, 1);
Number_A1_A2_crossings = zeros(2, 1);

All_A2_A3_crossings = strings(2, 1);
All_A1_A2_crossings = strings(2, 1);

Envelope_action_sequence = strings(2, 1);
Expected_A3_A2_A1_sequence = false(2, 1);

for ii = 1:2

    D = data_list{ii};
    diag_now = D.threshold_diagnostics;

    T1_true(ii) = D.T1_true;
    T2_true(ii) = D.T2_true;

    Number_A2_A3_crossings(ii) = ...
        diag_now.number_of_A2_A3_crossings;

    Number_A1_A2_crossings(ii) = ...
        diag_now.number_of_A1_A2_crossings;

    All_A2_A3_crossings(ii) = ...
        local_numeric_vector_to_string( ...
        diag_now.crossings_A2_A3);

    All_A1_A2_crossings(ii) = ...
        local_numeric_vector_to_string( ...
        diag_now.crossings_A1_A2);

    Envelope_action_sequence(ii) = ...
        string(diag_now.envelope_action_sequence);

    Expected_A3_A2_A1_sequence(ii) = ...
        diag_now.expected_monotonic_sequence;
end

Diagnostic_Table = table( ...
    chemistry_list, ...
    T1_true, ...
    T2_true, ...
    Number_A2_A3_crossings, ...
    All_A2_A3_crossings, ...
    Number_A1_A2_crossings, ...
    All_A1_A2_crossings, ...
    Envelope_action_sequence, ...
    Expected_A3_A2_A1_sequence, ...
    'VariableNames', { ...
    'Chemistry', ...
    'Selected_T1_true', ...
    'Selected_T2_true', ...
    'Number_of_A2_A3_crossings', ...
    'All_A2_A3_crossings_SOH', ...
    'Number_of_A1_A2_crossings', ...
    'All_A1_A2_crossings_SOH', ...
    'Envelope_action_sequence', ...
    'Expected_A3_to_A2_to_A1_sequence'});
end

function output_string = local_numeric_vector_to_string(values)
% Convert a numeric vector to a compact comma-separated string.

if isempty(values)

    output_string = "None";

else

    output_string = strjoin( ...
        compose('%.6f', values(:)), ', ');
end
end

function local_plot_action_value_panel(ax, D, panel_label, panel_title)
% Plot one chemistry-specific true-SOH action-value panel.

hold(ax, 'on');
box(ax, 'on');
grid(ax, 'on');

color_A1 = [0.00, 0.45, 0.74];
color_A2 = [0.47, 0.67, 0.19];
color_A3 = [0.85, 0.33, 0.10];

p1 = plot( ...
    ax, ...
    D.soh_grid, ...
    D.V_A1_plot, ...
    '-', ...
    'LineWidth', 2.0, ...
    'Color', color_A1, ...
    'DisplayName', 'A_1: Direct reuse');

p2 = plot( ...
    ax, ...
    D.soh_grid, ...
    D.V_A2_plot, ...
    '-', ...
    'LineWidth', 2.0, ...
    'Color', color_A2, ...
    'DisplayName', 'A_2: Refurbishment');

p3 = plot( ...
    ax, ...
    D.soh_grid, ...
    D.V_A3_plot, ...
    '-', ...
    'LineWidth', 2.0, ...
    'Color', color_A3, ...
    'DisplayName', 'A_3: Recycling');

all_values = [ ...
    D.V_A1_plot; ...
    D.V_A2_plot; ...
    D.V_A3_plot];

y_min = min(all_values);
y_max = max(all_values);
y_range = y_max - y_min;

if ~isfinite(y_range) || y_range <= 0
    y_range = 1;
end

if y_min > 0
    y_lower = 0;
else
    y_lower = y_min - 0.08 .* y_range;
end

y_upper = y_max + 0.12 .* y_range;

xlim(ax, [0.40, 1.00]);
ylim(ax, [y_lower, y_upper]);

xticks(ax, 0.4:0.1:1.0);

if isfinite(D.T1_true)

    xline( ...
        ax, ...
        D.T1_true, ...
        '--', ...
        'Color', [0.20, 0.35, 0.95], ...
        'LineWidth', 1.2, ...
        'HandleVisibility', 'off');

    text( ...
        ax, ...
        D.T1_true + 0.008, ...
        y_lower + 0.16 .* (y_upper - y_lower), ...
        sprintf('T_{1,true} = %.3f', D.T1_true), ...
        'FontName', 'Arial', ...
        'FontSize', 8.0, ...
        'Color', [0.20, 0.35, 0.95]);
end

if isfinite(D.T2_true)

    xline( ...
        ax, ...
        D.T2_true, ...
        '--', ...
        'Color', [0.95, 0.25, 0.25], ...
        'LineWidth', 1.2, ...
        'HandleVisibility', 'off');

    text( ...
        ax, ...
        D.T2_true - 0.145, ...
        y_lower + 0.82 .* (y_upper - y_lower), ...
        sprintf('T_{2,true} = %.3f', D.T2_true), ...
        'FontName', 'Arial', ...
        'FontSize', 8.0, ...
        'Color', [0.95, 0.25, 0.25]);
end

xlabel( ...
    ax, ...
    'True SOH', ...
    'FontName', 'Arial', ...
    'FontSize', 9);

ylabel( ...
    ax, ...
    'Action value (USD per pack)', ...
    'FontName', 'Arial', ...
    'FontSize', 9);

title( ...
    ax, ...
    panel_title, ...
    'FontName', 'Arial', ...
    'FontSize', 10, ...
    'FontWeight', 'normal');

legend( ...
    ax, ...
    [p1, p2, p3], ...
    {'A_1: Direct reuse', ...
     'A_2: Refurbishment', ...
     'A_3: Recycling'}, ...
    'Location', 'northwest', ...
    'Box', 'off', ...
    'FontName', 'Arial', ...
    'FontSize', 7.2);

text( ...
    ax, ...
    0.02, ...
    0.97, ...
    panel_label, ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', ...
    'FontName', 'Arial', ...
    'FontSize', 10, ...
    'FontWeight', 'bold');

set(ax, ...
    'FontName', 'Arial', ...
    'FontSize', 8.5, ...
    'LineWidth', 0.9, ...
    'TickDir', 'out', ...
    'GridLineStyle', ':', ...
    'GridAlpha', 0.25);
end

function SOH_data = local_extract_soh_from_mat( ...
    SOH_file, input_file, chemistry)
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

            day_count = double(SOH_file.day);

            if ~isscalar(day_count) || ...
                    ~isfinite(day_count) || ...
                    day_count < 1

                error('Invalid NMC day count in %s.', input_file);
            end

            day_count = min( ...
                floor(day_count), ...
                numel(SOH_raw));

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

    soh_col_idx = find( ...
        contains(lower(table_vars), 'soh'), ...
        1, 'first');

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

    error('No valid %s SOH values were found in %s.', ...
        chemistry, input_file);
end
end

function local_save_figure(fig_handle, fig_file, png_file, resolution)
% Save a figure as both FIG and PNG files.

fig_dir = fileparts(fig_file);

if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

savefig(fig_handle, fig_file);

exportgraphics( ...
    fig_handle, ...
    png_file, ...
    'Resolution', resolution);

fprintf('[Figure] Saved: %s\n', png_file);
end
