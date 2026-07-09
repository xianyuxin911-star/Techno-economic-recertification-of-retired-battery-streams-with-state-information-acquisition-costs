%% P1_Exp01_SourceDomain_QVPreprocessing.m
% Purpose: Preprocesses source-domain MAP files, extracts Step-7 Q-V features, saves the source-domain training dataset, and generates Supplementary Fig. 1.

clc; clear; close all;

%%  1. Configuration & Path Setup
cfg = struct('M', 512, ...
             'Nominal_Cap', 15.0, ...
             'Target_Step', 7, ...
             'qmin', 0.70, ...
             'qmax', 0.86, ...
             'save_fig', true, ...
             'save_png', true);

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end

part_dir = script_dir;

data_dir = fullfile(part_dir, 'Source_Data');
output_data_dir = fullfile(part_dir, 'Output', 'Data');
figure_sup_dir = fullfile(part_dir, 'Figures', 'Supplementary');

arrayfun(@(x) (exist(x{1}, 'dir') || mkdir(x{1})), {output_data_dir, figure_sup_dir});

if ~exist(data_dir, 'dir')
    error('Source data folder not found: %s', data_dir);
end

dataset_file = fullfile(output_data_dir, 'P1_SourceDomain_Dataset.mat');

sup_fig_file = fullfile(figure_sup_dir, 'SupFig01_SourceDomain_QVCurves.fig');
sup_png_file = fullfile(figure_sup_dir, 'SupFig01_SourceDomain_QVCurves.png');


%%  2. File Search & Initialization
files = dir(fullfile(data_dir, 'MAP*.mat'));
if isempty(files)
    files = dir(fullfile(data_dir, '*.mat'));
    files = files(~startsWith({files.name}, {'Final_', 'sourcedata'}));
end
if isempty(files), error('No source-domain files found in: %s', data_dir); end

num_files = length(files);
fprintf('Found %d source-domain files. Extracting Step-%d [%.2f, %.2f]...\n', ...
    num_files, cfg.Target_Step, cfg.qmin, cfg.qmax);

% Preallocate memory for speed
Train_X = zeros(num_files, cfg.M);
Train_Y = zeros(num_files, 1);
SN_List = cell(num_files, 1);
[QNorm_List, Voltage_List, File_ID_List] = deal(cell(num_files, 1));
SOH_List = nan(num_files, 1);

q_grid = linspace(cfg.qmin, cfg.qmax, cfg.M);
[count_skip_soh, count_no_step, count_invalid, valid_count] = deal(0);


%%   3. Data Processing Loop
for k = 1:num_files
    fn = files(k).name;
    
    try
        content = load(fullfile(files(k).folder, fn));
        % Flexibly extract table data
        if isfield(content, 'data'), rawT = content.data;
        elseif isfield(content, 'data_struct'), rawT = content.data_struct;
        else, count_invalid = count_invalid + 1; continue; end
        
        if isstruct(rawT), rawT = struct2table(rawT); end
        if ~istable(rawT), count_invalid = count_invalid + 1; continue; end

        cols = rawT.Properties.VariableNames;
        s_idx = find(contains(cols, 'Step') & ~contains(cols, 'time'), 1);
        c_idx = find(contains(cols, 'Capacity'), 1);
        v_idx = find(contains(cols, 'Voltage'), 1);

        if isempty(s_idx) || isempty(c_idx) || isempty(v_idx)
            count_invalid = count_invalid + 1; continue; 
        end

        idx_target = find(rawT.(cols{s_idx}) == cfg.Target_Step);
        if isempty(idx_target)
            count_no_step = count_no_step + 1; continue; 
        end

        V_raw = double(rawT.(cols{v_idx})(idx_target));
        Q_raw = double(rawT.(cols{c_idx})(idx_target));
        if numel(V_raw) < 10 || numel(Q_raw) < 10
            count_invalid = count_invalid + 1; continue; 
        end

        if max(Q_raw) > 100, Q_raw = Q_raw / 1000.0; end % mAh to Ah
        
        % SOH and Normalized Capacity Calculation
        Q_rel = Q_raw - min(Q_raw);
        SOH = max(Q_rel) / cfg.Nominal_Cap;
        Q_star = Q_rel / cfg.Nominal_Cap;

        % Store for visualization
        SOH_List(k) = SOH;
        QNorm_List{k} = Q_star;
        Voltage_List{k} = V_raw;
        File_ID_List{k} = fn(1:end-4);

        if SOH < cfg.qmax
            count_skip_soh = count_skip_soh + 1; continue; 
        end

        idx_valid = isfinite(Q_star) & isfinite(V_raw);
        Q_valid = Q_star(idx_valid);
        V_valid = V_raw(idx_valid);

        if numel(Q_valid) < 2
            count_invalid = count_invalid + 1; continue; 
        end

        % unique() automatically sorts the output, removing the need for sort()
        [Q_unique, ia] = unique(Q_valid, 'last');
        V_unique = V_valid(ia);

        if numel(Q_unique) < 2
            count_invalid = count_invalid + 1; continue; 
        end

        if Q_unique(1) > cfg.qmin + 1e-4
            Q_unique = [cfg.qmin; Q_unique];
            V_unique = [V_unique(1); V_unique];
        end

        % Interpolation and Storage
        valid_count = valid_count + 1;
        Train_X(valid_count, :) = interp1(Q_unique, V_unique, q_grid, 'linear', 'extrap');
        Train_Y(valid_count, 1) = SOH;
        SN_List{valid_count, 1} = fn(1:end-4);

        if mod(k, 50) == 0, fprintf('Processed %d/%d files...\n', k, num_files); end

    catch ME
        fprintf('Error in %s: %s\n', fn, ME.message);
    end
end

% Trim preallocated arrays
Train_X = Train_X(1:valid_count, :);
Train_Y = Train_Y(1:valid_count, :);
SN_List = SN_List(1:valid_count, :);


%%  4. Save Dataset & Summary
if isempty(Train_X), error('No valid samples generated. Check qmin/qmax.'); end
save(dataset_file, 'Train_X', 'Train_Y', 'SN_List', 'cfg');

fprintf('\n================ Preprocessing Completed ================\n');
fprintf('Feature window: [%.2f, %.2f] | Feature length: %d\n', cfg.qmin, cfg.qmax, cfg.M);
fprintf('Valid: %d | Skipped (Low SOH): %d | No Step: %d | Invalid: %d\n', ...
    valid_count, count_skip_soh, count_no_step, count_invalid);
fprintf('Saved dataset: %s\n', dataset_file);

%%  5. Visualize Q-V Curves
valid_soh = SOH_List(isfinite(SOH_List));
if isempty(valid_soh)
    warning('No valid SOH values found for visualization.');
    return; 
end

soh_min = min(valid_soh);
soh_max = max(valid_soh);

figure('Position', [200, 200, 900, 650], 'Color', 'w');
hold on; grid on; box on; set(gca, 'FontName', 'Arial', 'FontSize', 12);

cmap = turbo(256);
[~, sort_idx] = sort(SOH_List, 'ascend');

for kk = 1:length(sort_idx)
    idx = sort_idx(kk);
    current_soh = SOH_List(idx);
    
    if isnan(current_soh) || isempty(QNorm_List{idx}), continue; end
    
    % Color mapping calculation
    c_idx = floor(((current_soh - soh_min) / (soh_max - soh_min + 1e-6)) * 255) + 1;
    
    plot(QNorm_List{idx}, Voltage_List{idx}, ...
        'Color', cmap(max(1, min(256, c_idx)), :), 'LineWidth', 1.2);
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