%% P5_Exp06_FixedKBenchmark_PerPack.m
%
% Purpose:
% This script generates per-pack fixed-k benchmark plots for both LFP and NMC
% scale-dependent optimization results.
%
% It compares five fixed label-size policies, k = 5, 15, 25, 35, and 50,
% against the scale-dependent optimized policy under each deployment scale.
%

clear; clc; close all;
set(groot, 'DefaultFigureVisible', 'on');

%% 0. Path configuration

cfg = struct();

script_dir = fileparts(mfilename('fullpath'));

if isempty(script_dir)
 script_dir = pwd;
end

cfg.output_dir = fullfile(script_dir, 'Output');
cfg.results_dir = fullfile(cfg.output_dir, 'Results');

cfg.figure_root = fullfile(script_dir, 'Figures');
cfg.main_fig_dir = fullfile(cfg.figure_root, 'Main');
cfg.extended_fig_dir = fullfile(cfg.figure_root, 'Extended');

% Input files.
cfg.lfp_result_primary = fullfile(cfg.results_dir, ...
 'P5_Exp01_ScaleDependent_LFP_Results.mat');

cfg.lfp_result_legacy = fullfile(cfg.output_dir, ...
 'Result01_ScaleEffect_LFP_Results.mat');

cfg.nmc_result_primary = fullfile(cfg.results_dir, ...
 'P5_Exp02_ScaleDependent_NMC_Results.mat');

cfg.nmc_result_legacy = fullfile(cfg.output_dir, ...
 'Result02_ScaleEffect_NMC_Results.mat');

% Output tables and data.
cfg.lfp_table_csv = fullfile(cfg.results_dir, ...
 'P5_Exp06_LFP_FixedKBenchmark_PerPack_Table.csv');

cfg.nmc_table_csv = fullfile(cfg.results_dir, ...
 'P5_Exp06_NMC_FixedKBenchmark_PerPack_Table.csv');

cfg.output_data_mat = fullfile(cfg.results_dir, ...
 'P5_Exp06_FixedKBenchmark_PerPack_Data.mat');

% Output figures.
cfg.lfp_fig = fullfile(cfg.extended_fig_dir, ...
 'ExD06e_LFP_FixedKBenchmark_PerPack.fig');

cfg.lfp_png = fullfile(cfg.extended_fig_dir, ...
 'ExD06e_LFP_FixedKBenchmark_PerPack.png');

cfg.nmc_fig = fullfile(cfg.extended_fig_dir, ...
 'ExD06f_NMC_FixedKBenchmark_PerPack.fig');

cfg.nmc_png = fullfile(cfg.extended_fig_dir, ...
 'ExD06f_NMC_FixedKBenchmark_PerPack.png');

cfg.png_resolution = 600;

%% 1. User settings

benchmark_k = [5; 15; 25; 35; 50];

benchmark_definition = ...
 'Fixed k with T1 and T2 re-optimized conditional on k';

cost_tolerance_USD = 1e-7;

save_outputs = true;

line_width = 1;
marker_size = 4;
marker_list = {'o', 's', '^', 'd', 'v'};

line_alpha = 0.60;
marker_edge_alpha = 0.75;
marker_face_alpha = 0.25;

color_list = [ ...
 0.0000, 0.4470, 0.7410; ...
 0.8500, 0.3250, 0.0980; ...
 0.9290, 0.6940, 0.1250; ...
 0.4940, 0.1840, 0.5560; ...
 0.4660, 0.6740, 0.1880];

%% 2. Create output folders

folder_list = {cfg.results_dir, cfg.figure_root, cfg.extended_fig_dir};

for ii = 1:numel(folder_list)

 if ~exist(folder_list{ii}, 'dir')
 mkdir(folder_list{ii});
 end
end

%% 3. Process LFP and NMC

fprintf('================ P5 Exp06 fixed-k per-pack benchmark ================\n');

LFP = processFixedKPerPackBenchmark( ...
 'LFP', ...
 cfg.lfp_result_primary, ...
 cfg.lfp_result_legacy, ...
 benchmark_k);

NMC = processFixedKPerPackBenchmark( ...
 'NMC', ...
 cfg.nmc_result_primary, ...
 cfg.nmc_result_legacy, ...
 benchmark_k);

P5_Exp06_LFP_FixedKBenchmark_PerPack_Table = LFP.table;
P5_Exp06_NMC_FixedKBenchmark_PerPack_Table = NMC.table;

%% 4. Plot LFP per-pack benchmark
fig_lfp = plotPerPackBenchmark( ...
 LFP, ...
 'Extended Data Fig. 6e - LFP fixed-k policy regret per pack', ...
 marker_list, ...
 color_list, ...
 line_width, ...
 marker_size, ...
 line_alpha, ...
 marker_edge_alpha, ...
 marker_face_alpha);

%% 5. Plot NMC per-pack benchmark
fig_nmc = plotPerPackBenchmark( ...
 NMC, ...
 'Extended Data Fig. 6f - NMC fixed-k policy regret per pack', ...
 marker_list, ...
 color_list, ...
 line_width, ...
 marker_size, ...
 line_alpha, ...
 marker_edge_alpha, ...
 marker_face_alpha);

%% 6. Save outputs

if save_outputs

 writetable(P5_Exp06_LFP_FixedKBenchmark_PerPack_Table, cfg.lfp_table_csv);
 writetable(P5_Exp06_NMC_FixedKBenchmark_PerPack_Table, cfg.nmc_table_csv);

 save(cfg.output_data_mat, ...
 'cfg', ...
 'benchmark_k', ...
 'benchmark_definition', ...
 'cost_tolerance_USD', ...
 'LFP', ...
 'NMC', ...
 'P5_Exp06_LFP_FixedKBenchmark_PerPack_Table', ...
 'P5_Exp06_NMC_FixedKBenchmark_PerPack_Table');

 savefig(fig_lfp, cfg.lfp_fig);
 exportgraphics(fig_lfp, cfg.lfp_png, ...
 'Resolution', cfg.png_resolution, ...
 'BackgroundColor', 'white');

 savefig(fig_nmc, cfg.nmc_fig);
 exportgraphics(fig_nmc, cfg.nmc_png, ...
 'Resolution', cfg.png_resolution, ...
 'BackgroundColor', 'white');

 fprintf('\nSaved LFP table: %s\n', cfg.lfp_table_csv);
 fprintf('Saved NMC table: %s\n', cfg.nmc_table_csv);
 fprintf('Saved combined data: %s\n', cfg.output_data_mat);

 fprintf('Saved LFP figure FIG: %s\n', cfg.lfp_fig);
 fprintf('Saved LFP figure PNG: %s\n', cfg.lfp_png);

 fprintf('Saved NMC figure FIG: %s\n', cfg.nmc_fig);
 fprintf('Saved NMC figure PNG: %s\n', cfg.nmc_png);
else
 fprintf('\nPreview only. No files were saved.\n');
end

%% 7. Print key values

printPerPackSummary(LFP);
printPerPackSummary(NMC);

fprintf('================ P5 Exp06 fixed-k per-pack benchmark completed ================\n');

%% Local functions

function out = processFixedKPerPackBenchmark(chemistry, primary_file, legacy_file, benchmark_k)

 fprintf('\n================ %s fixed-k per-pack benchmark ================\n', chemistry);
 if isfile(primary_file)
 input_file = primary_file;
 elseif isfile(legacy_file)
 input_file = legacy_file;
 else

 error(['%s scale-dependent result file not found.\n' ...
 'Checked:\n%s\n%s\n'], ...
 chemistry, primary_file, legacy_file);
 end
 fprintf('Input file: %s\n', input_file);

 S = load(input_file);
 R = selectResultSource(S, chemistry);

 [scale_ton_list, scale_name] = readVectorByCandidates(R, getScaleCandidates(chemistry), ...
 sprintf('%s scale-ton list', chemistry));

 [k_vals, k_name] = readVectorByCandidates(R, {'k_vals', 'k_list', 'k_grid', 'k_values'}, ...
 sprintf('%s k values', chemistry));

 [N_list, N_name] = readVectorByCandidates(R, getPackCandidates(chemistry), ...
 sprintf('%s pack-number list', chemistry));

 [Total_Cost_NK, cost_name] = readMatrixByCandidates(R, getCostCandidates(chemistry), ...
 sprintf('%s scale-by-k total cost matrix', chemistry));

 scale_ton_list = double(scale_ton_list(:));
 k_vals = double(k_vals(:));
 N_list = double(N_list(:));
 Total_Cost_NK = double(squeeze(Total_Cost_NK));

 n_scale = numel(scale_ton_list);
 n_k = numel(k_vals);
 n_benchmark = numel(benchmark_k);

 Total_Cost_NK = orientCostMatrix(Total_Cost_NK, n_scale, n_k, chemistry, cost_name);

 if numel(N_list) ~= n_scale
 error('%s N_list length is inconsistent with scale_ton_list.', chemistry);
 end

 % Use the row-wise minimum of the saved scale-by-k matrix as the
 % authoritative optimized policy.
 [opt_total_cost, opt_k_idx] = min(Total_Cost_NK, [], 2);
 opt_k = k_vals(opt_k_idx);
 opt_cost_per_pack = opt_total_cost ./ N_list;

 % Saved optimum vectors are diagnostics only.
 [saved_opt_total_cost, saved_cost_source] = ...
 readOptionalVectorByCandidates(R, getOptCostCandidates(chemistry));

 [saved_opt_k, saved_k_source] = ...
 readOptionalVectorByCandidates(R, getOptKCandidates(chemistry));

 max_cost_diff = NaN;

 if ~isempty(saved_opt_total_cost)
 saved_opt_total_cost = double(saved_opt_total_cost(:));

 if numel(saved_opt_total_cost) ~= n_scale
 error('%s saved optimized total-cost vector has an inconsistent length.', chemistry);
 end

 max_cost_diff = max(abs(saved_opt_total_cost - opt_total_cost));
 cost_ref = max(1, max(abs(opt_total_cost)));

 if max_cost_diff > 1e-8 * cost_ref
 error(['%s saved optimized costs do not match the row-wise ' ...
 'minimum of Total_Cost_NK. Rerun the upstream optimization.'], ...
 chemistry);
 end
 end

 if ~isempty(saved_opt_k)
 saved_opt_k = double(saved_opt_k(:));

 if numel(saved_opt_k) ~= n_scale
 error('%s saved optimized-k vector has an inconsistent length.', chemistry);
 end

 if any(saved_opt_k ~= opt_k)
 error(['%s saved optimized-k sequence does not match the ' ...
 'row-wise minimizer of Total_Cost_NK.'], chemistry);
 end
 end

 fprintf('Scale variable: %s\n', scale_name);
 fprintf('Pack variable: %s\n', N_name);
 fprintf('k variable: %s\n', k_name);
 fprintf('Cost matrix: %s\n', cost_name);
 fprintf('Saved optimized-cost source: %s\n', saved_cost_source);
 fprintf('Saved optimized-k source: %s\n', saved_k_source);
 fprintf('Max saved-versus-derived optimized cost difference = %.6g USD\n', ...
 max_cost_diff);

 benchmark_cost = zeros(n_scale, n_benchmark);
 benchmark_cost_per_pack = zeros(n_scale, n_benchmark);
 additional_cost = zeros(n_scale, n_benchmark);
 additional_cost_per_pack = zeros(n_scale, n_benchmark);
 additional_cost_kUSD = zeros(n_scale, n_benchmark);
 additional_cost_million = zeros(n_scale, n_benchmark);
 relative_saving_percent = zeros(n_scale, n_benchmark);
 benchmark_k_idx = zeros(n_benchmark, 1);

 for bb = 1:n_benchmark

 k_ref = benchmark_k(bb);

 idx_now = find(k_vals == k_ref, 1, 'first');

 if isempty(idx_now)
 error('%s benchmark k = %.0f is not available in k_vals.', chemistry, k_ref);
 end

 benchmark_k_idx(bb) = idx_now;

 benchmark_cost(:, bb) = Total_Cost_NK(:, idx_now);
 benchmark_cost_per_pack(:, bb) = benchmark_cost(:, bb) ./ N_list;

 additional_cost(:, bb) = benchmark_cost(:, bb) - opt_total_cost;
 additional_cost_per_pack(:, bb) = additional_cost(:, bb) ./ N_list;

 additional_cost_kUSD(:, bb) = additional_cost(:, bb) ./ 1000;
 additional_cost_million(:, bb) = additional_cost(:, bb) ./ 1e6;

 relative_saving_percent(:, bb) = ...
 100 .* additional_cost(:, bb) ./ max(benchmark_cost(:, bb), eps);
 end

 cost_tolerance_USD = 1e-7;

 if any(additional_cost(:) < -cost_tolerance_USD)
 error(['%s has a fixed-k benchmark cost below the row-wise ' ...
 'optimized cost. Check Total_Cost_NK and its k-axis ordering.'], ...
 chemistry);
 end

 additional_cost(additional_cost < 0 & ...
 additional_cost >= -cost_tolerance_USD) = 0;

 additional_cost_per_pack = additional_cost ./ N_list;
 additional_cost_kUSD = additional_cost ./ 1000;
 additional_cost_million = additional_cost ./ 1e6;

 relative_saving_percent = ...
 100 .* additional_cost ./ max(benchmark_cost, eps);

 additional_cost_per_pack(abs(additional_cost_per_pack) < 1e-12) = 0;
 additional_cost_kUSD(abs(additional_cost_kUSD) < 1e-12) = 0;
 additional_cost_million(abs(additional_cost_million) < 1e-12) = 0;
 relative_saving_percent(abs(relative_saving_percent) < 1e-10) = 0;

 table_out = buildPerPackTable( ...
 chemistry, ...
 scale_ton_list, ...
 N_list, ...
 opt_k, ...
 opt_total_cost, ...
 opt_cost_per_pack, ...
 benchmark_k, ...
 benchmark_cost, ...
 benchmark_cost_per_pack, ...
 additional_cost, ...
 additional_cost_kUSD, ...
 additional_cost_million, ...
 additional_cost_per_pack, ...
 relative_saving_percent);

 out = struct();
 out.chemistry = chemistry;
 out.input_file = input_file;

 out.scale_ton_list = scale_ton_list;
 out.N_list = N_list;
 out.k_vals = k_vals;
 out.benchmark_k = benchmark_k(:);
 out.benchmark_k_idx = benchmark_k_idx;

 out.Total_Cost_NK = Total_Cost_NK;

 out.opt_total_cost = opt_total_cost;
 out.opt_cost_per_pack = opt_cost_per_pack;
 out.opt_k = opt_k;

 out.benchmark_cost = benchmark_cost;
 out.benchmark_cost_per_pack = benchmark_cost_per_pack;

 out.additional_cost = additional_cost;
 out.additional_cost_kUSD = additional_cost_kUSD;
 out.additional_cost_million = additional_cost_million;
 out.additional_cost_per_pack = additional_cost_per_pack;
 out.relative_saving_percent = relative_saving_percent;

 out.table = table_out;

 fprintf('Finished %s fixed-k per-pack benchmark.\n', chemistry);
end

function R = selectResultSource(S, chemistry)

 R = S;
 field_names = fieldnames(S);
 for ii = 1:numel(field_names)
 field_now = field_names{ii};
 if isstruct(S.(field_now))
 candidate = S.(field_now);
 has_scale = any(isfield(candidate, getScaleCandidates(chemistry)));
 has_cost = any(isfield(candidate, getCostCandidates(chemistry)));
 if has_scale && has_cost
 R = candidate;
 fprintf('Using result structure: %s\n', field_now);
 return;
 end
 end
 end
end

function candidates = getScaleCandidates(chemistry)

 if strcmpi(chemistry, 'LFP')
 candidates = {'scale_ton_list_LFP', 'scale_ton_list', 'ton_list', 'scale_list_ton'};
 else
 candidates = {'scale_ton_list_NMC', 'scale_ton_list', 'ton_list', 'scale_list_ton'};
 end
end

function candidates = getPackCandidates(chemistry)

 if strcmpi(chemistry, 'LFP')
 candidates = {'scale_pack_list_LFP', 'N_list_LFP', 'N_list', ...
 'scale_pack_list', 'pack_number_list'};
 else
 candidates = {'N_list_NMC', 'scale_pack_list_NMC', 'N_list', ...
 'scale_pack_list', 'pack_number_list'};
 end
end

function candidates = getCostCandidates(chemistry)

 if strcmpi(chemistry, 'LFP')
 candidates = {'Total_Cost_NK_LFP', 'Total_Cost_NK', ...
 'total_cost_NK', 'TotalCost_NK'};
 else
 candidates = {'Total_Cost_NK_NMC', 'Total_Cost_NK', ...
 'total_cost_NK', 'TotalCost_NK'};
 end
end

function candidates = getOptCostCandidates(chemistry)

 if strcmpi(chemistry, 'LFP')
 candidates = {'opt_total_cost_LFP', 'opt_total_cost', ...
 'opt_Total_Cost_LFP', 'opt_Total_Cost'};
 else
 candidates = {'opt_total_cost_NMC', 'opt_total_cost', ...
 'opt_Total_Cost_NMC', 'opt_Total_Cost'};
 end
end

function candidates = getOptKCandidates(chemistry)

 if strcmpi(chemistry, 'LFP')
 candidates = {'opt_k_LFP', 'opt_k', 'opt_k_list'};
 else
 candidates = {'opt_k_NMC', 'opt_k', 'opt_k_list'};
 end
end

function [v, var_name] = readVectorByCandidates(R, candidates, description)

 [v, var_name] = readOptionalVectorByCandidates(R, candidates);

 if isempty(v)
 disp('Available variables:');
 disp(fieldnames(R));
 error('Required vector is missing: %s', description);
 end
end

function [v, var_name] = readOptionalVectorByCandidates(R, candidates)

 v = [];
 var_name = '';

 for ii = 1:numel(candidates)

 name_i = candidates{ii};

 if isfield(R, name_i)

 value_i = R.(name_i);

 if isnumeric(value_i) && isvector(value_i)
 v = value_i(:);
 var_name = name_i;
 return;
 end
 end
 end
end

function [M, var_name] = readMatrixByCandidates(R, candidates, description)

 M = [];
 var_name = '';

 for ii = 1:numel(candidates)

 name_i = candidates{ii};

 if isfield(R, name_i)

 value_i = squeeze(R.(name_i));

 if isnumeric(value_i) && ismatrix(value_i)
 M = value_i;
 var_name = name_i;
 return;
 end
 end
 end

 disp('Available variables:');
 disp(fieldnames(R));
 error('Required matrix is missing: %s', description);
end

function C = orientCostMatrix(C_raw, n_scale, n_k, chemistry, cost_name)

 [n_row, n_col] = size(C_raw);

 if n_row == n_scale && n_col == n_k
 C = C_raw;
 fprintf('Cost matrix orientation for %s: scale x k\n', chemistry);

 elseif n_row == n_k && n_col == n_scale
 C = C_raw.';
 fprintf('Cost matrix orientation for %s: k x scale, transposed to scale x k\n', chemistry);
 else

 error(['Unexpected %s cost matrix "%s" size: [%d, %d]. ' ...
 'Expected [%d, %d] or [%d, %d].'], ...
 chemistry, cost_name, n_row, n_col, n_scale, n_k, n_k, n_scale);
 end
end

function table_out = buildPerPackTable( ...
 chemistry, ...
 scale_ton_list, ...
 N_list, ...
 opt_k, ...
 opt_total_cost, ...
 opt_cost_per_pack, ...
 benchmark_k, ...
 benchmark_cost, ...
 benchmark_cost_per_pack, ...
 additional_cost, ...
 additional_cost_kUSD, ...
 additional_cost_million, ...
 additional_cost_per_pack, ...
 relative_saving_percent)

 n_scale = numel(scale_ton_list);
 n_benchmark = numel(benchmark_k);

 Chemistry_col = repmat({chemistry}, n_scale * n_benchmark, 1);
 Scale_ton_col = repmat(scale_ton_list, n_benchmark, 1);
 Pack_number_col = repmat(N_list, n_benchmark, 1);
 Optimized_k_col = repmat(opt_k, n_benchmark, 1);
 Optimized_total_cost_col = repmat(opt_total_cost, n_benchmark, 1);
 Optimized_cost_per_pack_col = repmat(opt_cost_per_pack, n_benchmark, 1);

 Reference_k_col = zeros(n_scale * n_benchmark, 1);
 Reference_total_cost_col = zeros(n_scale * n_benchmark, 1);
 Reference_cost_per_pack_col = zeros(n_scale * n_benchmark, 1);
 Additional_cost_col = zeros(n_scale * n_benchmark, 1);
 Additional_cost_kUSD_col = zeros(n_scale * n_benchmark, 1);
 Additional_cost_million_col = zeros(n_scale * n_benchmark, 1);
 Additional_cost_per_pack_col = zeros(n_scale * n_benchmark, 1);
 Relative_saving_percent_col = zeros(n_scale * n_benchmark, 1);

 row_start = 1;

 for bb = 1:n_benchmark

 row_end = row_start + n_scale - 1;

 Reference_k_col(row_start:row_end) = benchmark_k(bb);
 Reference_total_cost_col(row_start:row_end) = benchmark_cost(:, bb);
 Reference_cost_per_pack_col(row_start:row_end) = benchmark_cost_per_pack(:, bb);
 Additional_cost_col(row_start:row_end) = additional_cost(:, bb);
 Additional_cost_kUSD_col(row_start:row_end) = additional_cost_kUSD(:, bb);
 Additional_cost_million_col(row_start:row_end) = additional_cost_million(:, bb);
 Additional_cost_per_pack_col(row_start:row_end) = additional_cost_per_pack(:, bb);
 Relative_saving_percent_col(row_start:row_end) = relative_saving_percent(:, bb);

 row_start = row_end + 1;
 end

 table_out = table( ...
 Chemistry_col, ...
 Scale_ton_col, ...
 Pack_number_col, ...
 Optimized_k_col, ...
 Optimized_total_cost_col, ...
 Optimized_cost_per_pack_col, ...
 Reference_k_col, ...
 Reference_total_cost_col, ...
 Reference_cost_per_pack_col, ...
 Additional_cost_col, ...
 Additional_cost_kUSD_col, ...
 Additional_cost_million_col, ...
 Additional_cost_per_pack_col, ...
 Relative_saving_percent_col, ...
 'VariableNames', { ...
 'Chemistry', ...
 'Scale_ton', ...
 'Pack_number', ...
 'Optimized_k', ...
 'Optimized_total_cost_USD', ...
 'Optimized_cost_per_pack_USD', ...
 'Reference_k', ...
 'Reference_total_cost_USD', ...
 'Reference_cost_per_pack_USD', ...
 'Additional_cost_USD', ...
 'Additional_cost_kUSD', ...
 'Additional_cost_million_USD', ...
 'Additional_cost_per_pack_USD', ...
 'Relative_saving_percent'});
end

function fig = plotPerPackBenchmark( ...
 data, ...
 title_text, ...
 marker_list, ...
 color_list, ...
 line_width, ...
 marker_size, ...
 line_alpha, ...
 marker_edge_alpha, ...
 marker_face_alpha)

 fig = figure('Name', title_text, ...
 'Color', 'w', ...
 'Units', 'centimeters', ...
 'Position', [4, 4, 13, 9]);

 ax = axes(fig);
 hold(ax, 'on');

 for bb = 1:numel(data.benchmark_k)

 marker_id = mod(bb - 1, numel(marker_list)) + 1;
 color_id = mod(bb - 1, size(color_list, 1)) + 1;

 base_color = color_list(color_id, :);

 % Use faded RGB colors to mimic line transparency.
 line_color = blendWithWhite(base_color, line_alpha);
 marker_edge_color = blendWithWhite(base_color, marker_edge_alpha);
 marker_face_color = blendWithWhite(base_color, marker_face_alpha);

 % Draw line first.
 plot(ax, data.scale_ton_list, data.additional_cost_per_pack(:, bb), ...
 '-', ...
 'LineWidth', line_width, ...
 'Color', line_color, ...
 'DisplayName', sprintf('k = %d', data.benchmark_k(bb)));

 % Draw markers separately so marker transparency can be adjusted.
 scatter(ax, data.scale_ton_list, data.additional_cost_per_pack(:, bb), ...
 marker_size, ...
 marker_list{marker_id}, ...
 'MarkerEdgeColor', marker_edge_color, ...
 'MarkerFaceColor', marker_face_color, ...
 'MarkerEdgeAlpha', marker_edge_alpha, ...
 'MarkerFaceAlpha', marker_face_alpha, ...
 'LineWidth', 1.0, ...
 'HandleVisibility', 'off');
 end

 yline(ax, 0, ':', ...
 'LineWidth', 1.0, ...
 'HandleVisibility', 'off');

 set(ax, ...
 'XScale', 'log', ...
 'FontName', 'Arial', ...
 'FontSize', 10, ...
 'LineWidth', 0.9, ...
 'TickDir', 'out', ...
 'TickLength', [0.006, 0.006]);

 box(ax, 'on');
 grid(ax, 'on');

 % Use only three major x-axis ticks, consistent with the other
 % scale-dependent figures.
 set(ax, 'XTick', [10, 100, 1000]);
 set(ax, 'XTickLabel', {'10', '100', '1000'});

 xlabel(ax, 'Battery scale (ton)', ...
 'FontName', 'Arial', ...
 'FontSize', 11);

 ylabel(ax, 'Fixed-k policy regret (USD pack^{-1})', ...
 'FontName', 'Arial', ...
 'FontSize', 11);

 title(ax, title_text, ...
 'FontName', 'Arial', ...
 'FontSize', 11, ...
 'FontWeight', 'normal');

 legend(ax, ...
 'Location', 'best', ...
 'Box', 'off', ...
 'FontName', 'Arial', ...
 'FontSize', 9);

 y_max = max(data.additional_cost_per_pack(:));

 if y_max <= 0
 y_max = 1;
 end

 ylim(ax, [0, y_max * 1.12]);
 xlim(ax, [min(data.scale_ton_list), max(data.scale_ton_list)]);

 hold(ax, 'off');
end

function printPerPackSummary(data)

 fprintf('\n========== %s fixed-k per-pack benchmark key values ==========\n', data.chemistry);

 for bb = 1:numel(data.benchmark_k)

 [max_per_pack, idx_max] = max(data.additional_cost_per_pack(:, bb));

 fprintf('Fixed k = %d:\n', data.benchmark_k(bb));
 fprintf(' Maximum per-pack policy regret: %.4f USD/pack at %.0f tons\n', ...
 max_per_pack, data.scale_ton_list(idx_max));
 fprintf(' Per-pack policy regret at largest scale %.0f tons: %.4f USD/pack\n', ...
 data.scale_ton_list(end), data.additional_cost_per_pack(end, bb));
 end
end

function faded_color = blendWithWhite(base_color, alpha_value)

 alpha_value = max(0, min(1, alpha_value));
 white_color = [1, 1, 1];
 faded_color = alpha_value .* base_color + ...
 (1 - alpha_value) .* white_color;
end