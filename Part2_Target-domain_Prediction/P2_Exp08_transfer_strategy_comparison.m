%% P2_Exp08_transfer_strategy_comparison.m
% Purpose:
% This script compares target-domain transfer strategies for pack-level SOH
% prediction. It reads prediction results from P2_Exp02 to P2_Exp05,
% evaluates zero-shot, frozen-encoder, fine-tuning, and L2-SP fine-tuning
% strategies, and generates the transfer-strategy comparison figure.


clear; clc; close all;

%% 1) Configuration
cfg = struct();

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end

part_dir = script_dir;

cfg = struct();

cfg.output_dir = fullfile(part_dir, 'Output');
cfg.output_results_dir = fullfile(cfg.output_dir, 'Results');
cfg.figure_dir = fullfile(part_dir, 'Figures', 'Extended');

if ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

% Input prediction result files.
cfg.zero_csv = fullfile(cfg.output_results_dir, 'P2_Exp02_ZeroShot_MinWorst1_PackPredictions.csv');
cfg.frozen_csv = fullfile(cfg.output_results_dir, 'P2_Exp03_FrozenEncoder_PackPredictions.csv');
cfg.finetune_csv = fullfile(cfg.output_results_dir, 'P2_Exp04_FineTune_PackPredictions.csv');
cfg.l2sp_csv = fullfile(cfg.output_results_dir, 'P2_Exp05_FineTuneL2SP_PackPredictions.csv');
% Output result tables.
cfg.metrics_csv = fullfile(cfg.output_results_dir, 'P2_Exp08_TransferStrategy_Metrics.csv');
cfg.pred_long_csv = fullfile(cfg.output_results_dir, 'P2_Exp08_TransferStrategy_PredictionsLong.csv');

% Output figures.
cfg.fig_tracking = fullfile(cfg.figure_dir, 'ExD01b_TransferStrategyTracking.fig');
cfg.png_tracking = fullfile(cfg.figure_dir, 'ExD01b_TransferStrategyTracking.png');

cfg.fig_error = fullfile(cfg.figure_dir, 'P2_Exp08_TransferStrategyErrorBoxplot_diagnostic.fig');
cfg.png_error = fullfile(cfg.figure_dir, 'P2_Exp08_TransferStrategyErrorBoxplot_diagnostic.png');

cfg.fig_metrics = fullfile(cfg.figure_dir, 'P2_Exp08_TransferStrategyRMSESummary_diagnostic.fig');
cfg.png_metrics = fullfile(cfg.figure_dir, 'P2_Exp08_TransferStrategyRMSESummary_diagnostic.png');

cfg.png_resolution = 600;

if ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

if ~exist(cfg.figure_dir, 'dir')
    mkdir(cfg.figure_dir);
end

fprintf('================ P2 Exp08: Transfer-strategy comparison ================\n');

%% 2) Load prediction results
T_zero = local_read_required_table(cfg.zero_csv);

pack_idx = local_get_numeric_col(T_zero, {'Pack_Index', 'Sample_Index', 'Index'});
y_true = local_get_numeric_col(T_zero, {'True_Pack_SOH', 'True_SOH'});
y_zero = local_get_numeric_col(T_zero, {'Pred_MinWorst1_SOH', 'Pred_Bottom_1', 'Pred_Bottom1', 'Predicted_SOH'});

y_frozen = local_read_aligned_prediction( ...
    cfg.frozen_csv, ...
    pack_idx, ...
    {'Predicted_SOH', 'Pred_FrozenEncoder', 'Pred_Frozen_Encoder'});

y_ft = local_read_aligned_prediction( ...
    cfg.finetune_csv, ...
    pack_idx, ...
    {'Predicted_SOH', 'Pred_FineTune', 'Pred_Fine_Tune'});

y_l2sp = local_read_aligned_prediction( ...
    cfg.l2sp_csv, ...
    pack_idx, ...
    {'Predicted_SOH', 'Pred_L2SP', 'Pred_FineTuneL2SP', 'Pred_FineTune_L2SP'});

y_true = y_true(:);
y_zero = y_zero(:);
y_frozen = y_frozen(:);
y_ft = y_ft(:);
y_l2sp = y_l2sp(:);

n_pack = numel(y_true);

fprintf('Loaded prediction results for %d packs.\n', n_pack);
fprintf('  - Zero-shot:       %s\n', cfg.zero_csv);
fprintf('  - Frozen encoder:  %s\n', cfg.frozen_csv);
fprintf('  - Fine-tune:       %s\n', cfg.finetune_csv);
fprintf('  - Fine-tune L2-SP: %s\n', cfg.l2sp_csv);

%% 3) Calculate prediction errors and metrics
method_names = { ...
    'Zero-shot', ...
    'Frozen encoder', ...
    'Fine-tune', ...
    'Fine-tune + L2-SP'};

method_xticklabels = { ...
    'Zero-shot', ...
    sprintf('Frozen\nencoder'), ...
    'Fine-tune', ...
    sprintf('Fine-tune\n+ L2-SP')};

Y_pred = [y_zero, y_frozen, y_ft, y_l2sp];
Error_mat = Y_pred - y_true;

RMSE = sqrt(mean(Error_mat.^2, 1))';
MAE  = mean(abs(Error_mat), 1)';
MAPE = mean(abs(Error_mat ./ y_true), 1)' * 100;

R2 = zeros(4, 1);
Bias = zeros(4, 1);
Error_STD = zeros(4, 1);

for i = 1:4
    e = Error_mat(:, i);
    R2(i) = 1 - sum(e.^2) / sum((y_true - mean(y_true)).^2);
    Bias(i) = mean(e);
    Error_STD(i) = std(e);
end

Metrics_Table = table( ...
    method_names(:), ...
    RMSE, ...
    MAE, ...
    MAPE, ...
    R2, ...
    Bias, ...
    Error_STD, ...
    'VariableNames', { ...
    'Method', ...
    'RMSE', ...
    'MAE', ...
    'MAPE_percent', ...
    'R2', ...
    'Bias', ...
    'Error_STD'});

disp(' ');
disp('========== Transfer strategy metrics ==========');
disp(Metrics_Table);

writetable(Metrics_Table, cfg.metrics_csv);

%% 4) Save long-format prediction table
T_pred_long = table();

for i = 1:4
    T_tmp = table();
    T_tmp.Pack_Index = pack_idx;
    T_tmp.True_SOH = y_true;
    T_tmp.Predicted_SOH = Y_pred(:, i);
    T_tmp.Error = Error_mat(:, i);
    T_tmp.Abs_Error = abs(Error_mat(:, i));
    T_tmp.Method = repmat(string(method_names{i}), n_pack, 1);

    T_pred_long = [T_pred_long; T_tmp]; %#ok<AGROW>
end

writetable(T_pred_long, cfg.pred_long_csv);

%% 5) Set colors, transparency, and line styles
color_true   = [0.00, 0.00, 0.00];
color_zero   = [0.4660, 0.6740, 0.1880];
color_frozen = [0.4940, 0.1840, 0.5560];
color_ft     = [0.0000, 0.4470, 0.7410];
color_l2sp   = [0.8500, 0.3250, 0.0980];

colors = [
    color_zero;
    color_frozen;
    color_ft;
    color_l2sp
];

alpha_zero_pt   = 0.45;
alpha_frozen_pt = 0.50;
alpha_ft_pt     = 0.70;
alpha_l2sp_pt   = 0.95;

lw_true   = 3.2;
lw_zero   = 1.2;
lw_frozen = 1.2;
lw_ft     = 1.8;
lw_l2sp   = 2.4;

ms_zero   = 44;
ms_frozen = 44;
ms_ft     = 46;
ms_l2sp   = 72;

line_zero   = local_lighten(color_zero,   0.45);
line_frozen = local_lighten(color_frozen, 0.45);
line_ft     = local_lighten(color_ft,     0.28);
line_l2sp   = local_lighten(color_l2sp,   0.08);

%% 6) Panel C: SOH tracking
[y_true_sorted, sort_idx] = sort(y_true, 'descend');

y_zero_sorted   = y_zero(sort_idx);
y_frozen_sorted = y_frozen(sort_idx);
y_ft_sorted     = y_ft(sort_idx);
y_l2sp_sorted   = y_l2sp(sort_idx);

x_axis = 1:n_pack;

min_val = min([y_true; y_zero; y_frozen; y_ft; y_l2sp]) - 0.01;
max_val = max([y_true; y_zero; y_frozen; y_ft; y_l2sp]) + 0.01;

fig_tracking = figure( ...
    'Name', 'ExD01b_TransferStrategyTracking', ...
    'Color', 'w', ...
    'Position', [120, 120, 1050, 560]);

ax_main = axes('Position', [0.09, 0.13, 0.86, 0.78]);
hold(ax_main, 'on');
grid(ax_main, 'on');
box(ax_main, 'on');

set(ax_main, ...
    'GridLineStyle', ':', ...
    'GridAlpha', 0.45, ...
    'FontName', 'Arial', ...
    'FontSize', 12, ...
    'LineWidth', 1.2);

% True SOH.
h_true = plot(ax_main, x_axis, y_true_sorted, '-', ...
    'Color', color_true, ...
    'LineWidth', lw_true, ...
    'DisplayName', 'True SOH');

% Prediction trajectories.
plot(ax_main, x_axis, y_zero_sorted, '-', ...
    'Color', line_zero, ...
    'LineWidth', lw_zero, ...
    'HandleVisibility', 'off');

plot(ax_main, x_axis, y_frozen_sorted, '-', ...
    'Color', line_frozen, ...
    'LineWidth', lw_frozen, ...
    'HandleVisibility', 'off');

plot(ax_main, x_axis, y_ft_sorted, '-', ...
    'Color', line_ft, ...
    'LineWidth', lw_ft, ...
    'HandleVisibility', 'off');

plot(ax_main, x_axis, y_l2sp_sorted, '-', ...
    'Color', line_l2sp, ...
    'LineWidth', lw_l2sp, ...
    'HandleVisibility', 'off');

% Prediction markers.
h_zero = scatter(ax_main, x_axis, y_zero_sorted, ms_zero, color_zero, 'd', 'filled', ...
    'MarkerFaceAlpha', alpha_zero_pt, ...
    'MarkerEdgeAlpha', alpha_zero_pt, ...
    'DisplayName', 'Zero-shot');

h_frozen = scatter(ax_main, x_axis, y_frozen_sorted, ms_frozen, color_frozen, '^', 'filled', ...
    'MarkerFaceAlpha', alpha_frozen_pt, ...
    'MarkerEdgeAlpha', alpha_frozen_pt, ...
    'DisplayName', 'Frozen encoder');

h_ft = scatter(ax_main, x_axis, y_ft_sorted, ms_ft, color_ft, 's', 'filled', ...
    'MarkerFaceAlpha', alpha_ft_pt, ...
    'MarkerEdgeAlpha', alpha_ft_pt, ...
    'DisplayName', 'Fine-tune');

h_l2sp = scatter(ax_main, x_axis, y_l2sp_sorted, ms_l2sp, color_l2sp, 'o', 'filled', ...
    'MarkerFaceAlpha', alpha_l2sp_pt, ...
    'MarkerEdgeAlpha', alpha_l2sp_pt, ...
    'DisplayName', 'Fine-tune + L2-SP');

xlabel(ax_main, 'Sorted battery pack samples');
ylabel(ax_main, 'Predicted / true SOH');

xlim(ax_main, [1, n_pack]);
ylim(ax_main, [min_val, max_val]);

legend(ax_main, [h_true, h_zero, h_frozen, h_ft, h_l2sp], ...
    {'True SOH', 'Zero-shot', 'Frozen encoder', 'Fine-tune', 'Fine-tune + L2-SP'}, ...
    'Location', 'southwest', ...
    'Box', 'off', ...
    'FontSize', 10);

savefig(fig_tracking, cfg.fig_tracking);
exportgraphics(fig_tracking, cfg.png_tracking, 'Resolution', cfg.png_resolution);

%% 7) Panel D: Error distribution boxplot with scatter
fig_error = figure( ...
    'Name', 'P2_Exp08_TransferStrategyErrorBoxplot_diagnostic', ...
    'Color', 'w', ...
    'Position', [200, 200, 750, 500]);

hold on;
grid on;
box on;

h = boxplot(Error_mat, ...
    'Labels', method_names, ...
    'Widths', 0.50, ...
    'Symbol', '');

set(h, 'LineWidth', 1.2, 'Color', [0.3 0.3 0.3]);

boxes = findobj(gca, 'Tag', 'Box');

for j = 1:length(boxes)
    patch(get(boxes(j), 'XData'), get(boxes(j), 'YData'), ...
        colors(5-j, :), ...
        'FaceAlpha', 0.18, ...
        'EdgeColor', 'none');
end

rng(42);

for i = 1:4
    y = Error_mat(:, i);
    x_jitter = i + (rand(size(y)) - 0.5) * 0.16;

    scatter(x_jitter, y, 32, colors(i, :), 'filled', ...
        'MarkerFaceAlpha', 0.50, ...
        'MarkerEdgeColor', 'none');
end

yline(0, '--', ...
    'Color', [0.65 0 0], ...
    'LineWidth', 1.0, ...
    'HandleVisibility', 'off');

max_err = max(abs(Error_mat(:)));
ylim([-max_err * 1.25, max_err * 1.25]);
xlim([0.5, 4.5]);

ylabel('Prediction error (\DeltaSOH)');
title('Prediction-error distribution');

set(gca, ...
    'FontSize', 12, ...
    'FontName', 'Arial', ...
    'LineWidth', 1.2, ...
    'TickDir', 'out', ...
    'Box', 'on', ...
    'GridLineStyle', ':', ...
    'GridAlpha', 0.18);

ax = gca;
ax.XTick = 1:4;
ax.XTickLabel = method_xticklabels;

savefig(fig_error, cfg.fig_error);
exportgraphics(fig_error, cfg.png_error, 'Resolution', cfg.png_resolution);

%% 8) Panel E: RMSE lollipop chart
fig_metrics = figure( ...
    'Name', 'P2_Exp08_TransferStrategyRMSESummary_diagnostic', ...
    'Color', 'w', ...
    'Position', [280, 280, 620, 500]);

hold on;
grid on;
box on;

x_pos = 1:4;

% Light grey connecting line.
plot(x_pos, RMSE, ':', ...
    'Color', [0.55 0.55 0.55], ...
    'LineWidth', 1.3, ...
    'HandleVisibility', 'off');

% Lollipop stems and markers.
for i = 1:4

    line([i, i], [0, RMSE(i)], ...
        'Color', [0.60 0.60 0.60], ...
        'LineWidth', 2.0);

    marker_size = 160;

    if i == 4
        marker_size = 230;
    end

    scatter(i, RMSE(i), marker_size, colors(i, :), 'filled', ...
        'MarkerEdgeColor', [0.15 0.15 0.15], ...
        'LineWidth', 1.1);

    text(i, RMSE(i) + max(RMSE) * 0.055, sprintf('%.4f', RMSE(i)), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 11, ...
        'FontWeight', 'bold', ...
        'Color', colors(i, :), ...
        'FontName', 'Arial');
end

ylabel('RMSE');
title('Pack-level SOH prediction error');

set(gca, ...
    'XGrid', 'off', ...
    'YGrid', 'on', ...
    'GridLineStyle', ':', ...
    'GridColor', 'k', ...
    'GridAlpha', 0.15, ...
    'FontSize', 12, ...
    'FontName', 'Arial', ...
    'TickDir', 'out', ...
    'Box', 'on', ...
    'LineWidth', 1);

xlim([0.5, 4.5]);
ylim([0, max(RMSE) * 1.25]);

ax = gca;
ax.XTick = 1:4;
ax.XTickLabel = method_xticklabels;

savefig(fig_metrics, cfg.fig_metrics);
exportgraphics(fig_metrics, cfg.png_metrics, 'Resolution', cfg.png_resolution);

fprintf('\n>>> P2_Exp08 transfer-strategy comparison figures completed.\n');
fprintf('>>> Saved metrics:          %s\n', cfg.metrics_csv);
fprintf('>>> Saved long predictions: %s\n', cfg.pred_long_csv);
fprintf('>>> Saved panel C:          %s\n', cfg.png_tracking);
fprintf('>>> Saved panel D:          %s\n', cfg.png_error);
fprintf('>>> Saved panel E:          %s\n', cfg.png_metrics);


%% Local functions
function T = local_read_required_table(filename)
% Read a required CSV table.

if ~isfile(filename)
    error('File not found: %s', filename);
end

T = readtable(filename, 'VariableNamingRule', 'preserve');
end

function col = local_find_col(vars, candidates)
% Find a column by matching normalized column names.

col = '';

vars_norm = cell(size(vars));
for i = 1:numel(vars)
    vars_norm{i} = local_normalize_name(vars{i});
end

for i = 1:numel(candidates)
    cand_norm = local_normalize_name(candidates{i});
    idx = find(strcmp(vars_norm, cand_norm), 1);

    if ~isempty(idx)
        col = vars{idx};
        return;
    end
end
end

function s = local_normalize_name(s)
% Normalize column names for robust matching.

s = char(s);
s = lower(s);
s = regexprep(s, '[^a-z0-9]', '');
end

function x = local_get_numeric_col(T, candidates)
% Extract a numeric column from a table using candidate column names.

vars = T.Properties.VariableNames;
col = local_find_col(vars, candidates);

if isempty(col)
    disp(vars');
    error('Required column not found. Candidate names: %s', strjoin(candidates, ', '));
end

x = T.(col);

if iscell(x) || isstring(x) || ischar(x)
    x = str2double(string(x));
end

x = double(x);
x = x(:);
end

function y = local_read_aligned_prediction(filename, pack_idx_ref, pred_candidates)
% Read prediction values from a CSV file and align them by Pack_Index.

T = local_read_required_table(filename);

pred = local_get_numeric_col(T, pred_candidates);

vars = T.Properties.VariableNames;
idx_col = local_find_col(vars, {'Pack_Index', 'Sample_Index', 'Index'});

if isempty(idx_col)
    if height(T) ~= numel(pack_idx_ref)
        error('Pack_Index was not found in %s, and row numbers are inconsistent.', filename);
    end

    y = pred(:);
    return;
end

idx = local_get_numeric_col(T, {'Pack_Index', 'Sample_Index', 'Index'});

[tf, loc] = ismember(pack_idx_ref, idx);

if any(~tf)
    error('Some Pack_Index values from the zero-shot file were not found in %s.', filename);
end

y = pred(loc);
y = y(:);
end

function c_out = local_lighten(c_in, amount)
% Lighten an RGB color.

c_out = c_in + (1 - c_in) * amount;
c_out(c_out > 1) = 1;
end
