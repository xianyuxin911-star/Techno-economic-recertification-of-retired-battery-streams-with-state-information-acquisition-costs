%% P2_Exp01_target_data_processing.m
% Purpose: Constructs the target-domain pack-level input dataset from 001PB files,
% extracts fixed-length 45-channel Step-discharge voltage features, saves the
% target-domain dataset for downstream transfer learning, and generates the
% Supplementary target-domain Q-V curve figure.

clc; clear; close all;

%%  1. Configuration & Path Setup

cfg = struct('M', 512, 'Nominal_Cap', 176.0, 'I_dis_thr', 10, 'minPts', 60, ...
             'qmin', 0.70, 'qmax', 0.86, 'visualSOHLimit', 0.80, ...
             'show_window_patch', false, 'save_fig', true, 'save_png', true);

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end

part_dir = script_dir;

data_dir        = fullfile(part_dir, 'Target_Data');
output_data_dir = fullfile(part_dir, 'Output', 'Data');
figure_sup_dir  = fullfile(part_dir, 'Figures', 'Supplementary');

arrayfun(@(x) (exist(x{1}, 'dir') || mkdir(x{1})), ...
    {output_data_dir, figure_sup_dir});

if ~exist(data_dir, 'dir')
    error('Target data folder not found: %s', data_dir);
end

dataset_file = fullfile(output_data_dir, 'P2_Exp01_TargetPack_Input.mat');

sup_fig_file = fullfile(figure_sup_dir, 'SupFig01b_target_domain_QV_curves.fig');
sup_png_file = fullfile(figure_sup_dir, 'SupFig01b_target_domain_QV_curves.png');

%%  2. File Search & Memory Preallocation

files = dir(fullfile(data_dir, '001PB*.mat'));
if isempty(files), error('No target-domain 001PB*.mat files found.'); end

num_files = length(files);
fprintf('Found %d target-domain files. Extracting [%.2f, %.2f]...\n', num_files, cfg.qmin, cfg.qmax);

% Preallocate variables for speed (Crucial for 4D tensor X_img)
X_img   = zeros(45, cfg.M, 1, num_files, 'single');
y_all   = zeros(num_files, 1, 'single');
SN_List = strings(num_files, 1);
[QNorm_List, VPack_List] = deal(cell(num_files, 1));
SOH_List = nan(num_files, 1);
File_ID_List = strings(num_files, 1);

q_grid = linspace(cfg.qmin, cfg.qmax, cfg.M);
[count_ok, count_skip_depth, count_skip_visual_soh, count_no_step, count_invalid] = deal(0);

%%  3. Data Processing Loop

for k = 1:num_files
    fn = files(k).name;
    short_fn = erase(fn, ".mat");
    
    try
        S = load(fullfile(files(k).folder, fn));
        vars = fieldnames(S);
        data_cell = S.(vars{1});
        if ~iscell(data_cell), data_cell = num2cell(data_cell); end
        
        headers = string(data_cell(1, :));
        raw = data_cell(2:end, :);

        % Locate required columns
        idx_step = local_find_col(headers, ["步骤", "Step"]);
        idx_curr = local_find_col(headers, ["电流", "Current", "直流电流"]);
        idx_cap  = local_find_col(headers, ["放电容量", "DischargeCapacity", "Discharge", "容量", "Capacity"]);
        [col_indices, ~] = local_find_CAN_cellV(headers);

        if isempty(idx_step) || isempty(idx_curr) || isempty(idx_cap) || numel(col_indices) ~= 45
            count_invalid = count_invalid + 1; continue;
        end

        Steps = local_to_double(raw(:, idx_step));
        Currs = local_to_double(raw(:, idx_curr));
        Caps  = local_to_double(raw(:, idx_cap));

        % Select discharge step
        step_dis = local_pick_discharge_step(Steps, Currs, Caps, cfg.I_dis_thr);
        if isempty(step_dis), count_no_step = count_no_step + 1; continue; end

        idx_dis = (Steps == step_dis);
        if sum(idx_dis) < cfg.minPts, count_invalid = count_invalid + 1; continue; end

        % Process capacity & SOH
        Q_raw = Caps(idx_dis);
        if abs(Q_raw(end) - Q_raw(1)) > 5 * cfg.Nominal_Cap, Q_raw = Q_raw / 1000; end
        Q_base = Q_raw - Q_raw(1);
        if Q_base(end) < 0, Q_base = -Q_base; end
        
        Q_norm = Q_base / cfg.Nominal_Cap;
        SOH_pack = Q_base(end) / cfg.Nominal_Cap;

        % Store pack-level curves for visualization
        V_pack = mean(local_to_double(raw(idx_dis, col_indices)), 2, 'omitnan');
        SOH_List(k) = SOH_pack;
        QNorm_List{k} = Q_norm;
        VPack_List{k} = V_pack;
        File_ID_List(k) = short_fn;

        if SOH_pack < cfg.visualSOHLimit, count_skip_visual_soh = count_skip_visual_soh + 1; end
        if max(Q_norm) < cfg.qmax - 1e-4, count_skip_depth = count_skip_depth + 1; continue; end

        % Extract and resample 45 cell-voltage curves
        Vrs = nan(45, cfg.M);
        ok_pack = true;

        for i = 1:45
            V_raw = local_to_double(raw(idx_dis, col_indices(i)));
            good = isfinite(Q_norm) & isfinite(V_raw);
            Qv = Q_norm(good); Vv = V_raw(good);

            if numel(Qv) < 20, ok_pack = false; break; end

            % unique() inherently sorts the data, making sort() redundant
            [Q_u, ia] = unique(Qv, 'last');
            V_u = Vv(ia);
            
            if numel(Q_u) < 2, ok_pack = false; break; end

            Vr = interp1(Q_u, V_u, q_grid, 'linear', 'extrap');
            if any(isnan(Vr)), ok_pack = false; break; end
            Vrs(i, :) = Vr;
        end

        if ~ok_pack
            fprintf('  [SKIP] %s failed during interpolation.\n', fn); continue;
        end

        % Store valid pack sample
        count_ok = count_ok + 1;
        X_img(:, :, :, count_ok) = reshape(single(Vrs), [45, cfg.M, 1]);
        y_all(count_ok, 1) = single(SOH_pack);
        SN_List(count_ok, 1) = short_fn;

        if mod(count_ok, 10) == 0
            fprintf('  [Progress] Generated %d samples. Latest SOH = %.4f\n', count_ok, SOH_pack);
        end

    catch ME
        fprintf('  [Error] %s: %s\n', fn, ME.message);
    end
end

% Trim preallocated arrays
X_img   = X_img(:, :, :, 1:count_ok);
y_all   = y_all(1:count_ok);
SN_List = SN_List(1:count_ok);

%% 4. Save Dataset & Summary

if isempty(y_all), error('No valid pack samples generated. Check Qmax limit.'); end
save(dataset_file, 'X_img', 'y_all', 'SN_List', 'cfg', ...
     'QNorm_List', 'VPack_List', 'SOH_List', 'File_ID_List');

fprintf('\n================ Preprocessing Completed ================\n');
fprintf('Window: [%.2f, %.2f] | Valid packs: %d\n', cfg.qmin, cfg.qmax, count_ok);
fprintf('Skipped (Low depth): %d | No step: %d | Invalid: %d\n', count_skip_depth, count_no_step, count_invalid);
fprintf('Supplementary target-domain Q-V figure completed. Packs with SOH < %.2f excluded.\n', cfg.visualSOHLimit);

%% =========================================================
%  5. Visualize Q-V Curves
% =========================================================
visual_idx = isfinite(SOH_List) & SOH_List >= cfg.visualSOHLimit;
if ~any(visual_idx)
    warning('No valid target-domain packs available for visualization.');
    return;
end

soh_min = min(SOH_List(visual_idx));
soh_max = max(SOH_List(visual_idx));

figure('Position', [200, 200, 900, 650], 'Color', 'w');
hold on; grid on; box on; set(gca, 'FontName', 'Arial', 'FontSize', 12);

cmap = turbo(256);
[~, sort_idx] = sort(SOH_List, 'ascend');

for kk = 1:length(sort_idx)
    idx = sort_idx(kk);
    current_soh = SOH_List(idx);
    
    if isnan(current_soh) || current_soh < cfg.visualSOHLimit || isempty(QNorm_List{idx}), continue; end

    c_idx = floor(((current_soh - soh_min) / (soh_max - soh_min + 1e-6)) * 255) + 1;
    plot(QNorm_List{idx}, VPack_List{idx}, 'Color', cmap(max(1, min(256, c_idx)), :), 'LineWidth', 1.2);
end

xlabel('Normalized capacity, Q/Q_{nom}', 'FontSize', 12);
ylabel('Voltage (V)', 'FontSize', 12);
xlim([0, 1.05]); ylim([2.4, 3.4]);

cb = colorbar; cb.Label.String = 'SOH'; clim([soh_min, soh_max]);
drawnow;

if cfg.save_fig
    savefig(sup_fig_file);
    fprintf('Saved Supplementary figure: %s\n', sup_fig_file);
end

if cfg.save_png
    exportgraphics(gcf, sup_png_file, 'Resolution', 600);
    fprintf('Saved Supplementary PNG: %s\n', sup_png_file);
end

fprintf('Supplementary target-domain Q-V figure completed. Packs with SOH < %.2f excluded.\n', cfg.visualSOHLimit);

%%  Local  Functions

function idx = local_find_col(headers, keys)
    idx = []; h = lower(strrep(headers, " ", ""));
    for k = keys
        hit = find(contains(h, lower(strrep(k, " ", ""))), 1, 'first');
        if ~isempty(hit), idx = hit; return; end
    end
end

function x = local_to_double(col)
    if isnumeric(col), x = double(col); return; end
    x = str2double(strrep(strtrim(string(col)), ",", ""));
end

function step_dis = local_pick_discharge_step(Steps, Currs, Caps, Ithr)
    step_dis = []; best_dQ = -inf;
    for s = reshape(unique(Steps(isfinite(Steps))), 1, [])
        idx = (Steps == s); I_med = median(Currs(idx), 'omitnan');
        if ~isfinite(I_med) || I_med >= -Ithr, continue; end
        q = Caps(idx); q = q(isfinite(q));
        if numel(q) < 50, continue; end
        if abs(q(end) - q(1)) > best_dQ
            best_dQ = abs(q(end) - q(1)); step_dis = s;
        end
    end
end

function [col_indices, cell_names] = local_find_CAN_cellV(headers)
    h = string(headers);
    cols = find(~cellfun('isempty', regexp(cellstr(h), '^CAN[12]_V\d+$', 'once')));
    toks = regexp(h(cols), '^CAN(\d+)_V(\d+)$', 'tokens', 'once');
    canID = cellfun(@(x) str2double(x{1}), toks);
    vID = cellfun(@(x) str2double(x{2}), toks);
    [~, ord] = sortrows([canID(:), vID(:)], [1 2]);
    col_indices = cols(ord); cell_names = h(col_indices);
end