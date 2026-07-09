%% P2_Exp07_attention_pooling_ablation.m
% Purpose:
% This script compares cell-to-pack pooling and attention mechanisms for
% target-domain pack-level SOH prediction under the same L2-SP fine-tuning
% framework. Mean pooling, standard attention, soft-min attention, and a
% near-hard soft-min approximation are evaluated using leave-one-pack-out
% cross-validation.

clc; clear; close all;

%% Configuration

% Source-domain CNN model generated in Part1.
script_dir = fileparts(mfilename('fullpath'));

if isempty(script_dir), script_dir = pwd; end

part_dir = script_dir;
output_results_dir = fullfile(part_dir, 'Output', 'Results');

if ~exist(output_results_dir, 'dir')
    mkdir(output_results_dir);
end

cfg = struct();
cfg.experiment_id = 'P2_Exp07';
cfg.method = 'AttentionPoolingAblation';
cfg.target_mat = fullfile(part_dir, 'Output', 'Data', 'P2_Exp01_TargetPack_Input.mat');
cfg.source_mat = fullfile(part_dir, '..', 'Part1_Source-domain_Prediction', 'Output', 'Models', 'P1_SourceDomain_CNN_Model.mat');
cfg.metrics_csv = fullfile(output_results_dir, 'P2_Exp07_AttentionPoolingAblation_Metrics.csv');
cfg.result_mat = fullfile(output_results_dir, 'P2_Exp07_AttentionPoolingAblation_Results.mat');

% Input dimensions and network settings.
cfg.M = 512;
cfg.nCell = 45;
cfg.dEmbed = 64;
cfg.dAttH  = 32;
cfg.dHeadH = 16;

% Training settings.
cfg.maxEpochs = 60;
cfg.warmupEpochs = 20;
cfg.lr_head = 1e-3;
cfg.lr_enc  = 1e-5;
cfg.tune_last_conv = 3;
cfg.beta1 = 0.9;
cfg.beta2 = 0.999;

% Regularization and attention settings.
cfg.attn_tau  = 0.1;
cfg.l2_head   = 1e-4;
cfg.lambda_sp = 0.005;
cfg.clip_enc  = 1.0;
cfg.clip_head = 1.0;

if ~exist('Output', 'dir')
    mkdir('Output');
end

fprintf('================ P2_Exp07 attention-pooling ablation ================\n');
fprintf('Target-domain data file:  %s\n', cfg.target_mat);
fprintf('Source-domain model file: %s\n', cfg.source_mat);

%% 1) Load target-domain data and source-domain model
if ~exist(cfg.target_mat, 'file')
    error('Target-domain data not found: %s', cfg.target_mat);
end

if ~exist(cfg.source_mat, 'file')
    error('Source-domain model not found: %s', cfg.source_mat);
end

T = load(cfg.target_mat);

X_img = single(T.X_img);
y_all = single(T.y_all(:));

if isfield(T, 'SN_List')
    SN_List = T.SN_List(:);
else
    SN_List = strings(numel(y_all), 1);
    for i = 1:numel(y_all)
        SN_List(i) = sprintf('Pack_%d', i);
    end
end

if isfield(T, 'cfg')
    cfgT = T.cfg;
else
    cfgT = struct();
end

Npack = size(X_img, 4);

S = load(cfg.source_mat);

netFull = S.netFull;
muX = double(S.muX_full);
sgX = double(S.sgX_full);

if sgX < 1e-12
    sgX = 1;
end

% Normalize target-domain voltage curves using source-domain statistics.
X_img = (X_img - muX) / sgX;

% Prepare the reference encoder for L2-SP regularization.
layers = netFull.Layers;
idx_gap = find(arrayfun(@(L) strcmp(L.Name, 'gap'), layers), 1, 'first');
lgraphEnc = layerGraph(layers(1:idx_gap));
enc_ref = dlnetwork(lgraphEnc);

fprintf('[Target] Dataset loaded successfully | Npack = %d\n', Npack);

%% 2) Define pooling and attention mechanisms for comparison
pool_methods = {'Mean', 'Standard', 'Softmin', 'Hardmin'};
num_methods = length(pool_methods);

% Store all prediction, attention, and metric results.
Results = struct();

%% 3) Run leave-one-pack-out experiments for each mechanism
for m = 1:num_methods

    current_method = pool_methods{m};

    fprintf('\n======================================================\n');
    fprintf('Starting experiment [%d/%d]: %s pooling\n', ...
        m, num_methods, current_method);
    fprintf('======================================================\n');

    rng(42);

    yhat = nan(Npack, 1, 'single');
    MaxAttn_List = nan(Npack, 1);
    Cell_Attn_All = zeros(cfg.nCell, Npack);

    % Leave-one-pack-out training.
    for te = 1:Npack

        tr = setdiff(1:Npack, te);

        y_train_raw = y_all(tr);
        muY = mean(y_train_raw);
        sgY = std(y_train_raw);

        if sgY < 1e-12
            sgY = 1;
        end

        y_train_norm = (y_train_raw - muY) / sgY;

        % Reset the trainable encoder for the current fold.
        currEnc = dlnetwork(lgraphEnc);

        % Initialize attention and regression-head parameters.
        attW1 = dlarray(0.01 * randn(cfg.dAttH, cfg.dEmbed, 'single'));
        attb1 = dlarray(zeros(cfg.dAttH, 1, 'single'));
        attW2 = dlarray(0.01 * randn(1, cfg.dAttH, 'single'));
        attb2 = dlarray(zeros(1, 1, 'single'));

        headW1 = dlarray(0.01 * randn(cfg.dHeadH, cfg.dEmbed, 'single'));
        headb1 = dlarray(zeros(cfg.dHeadH, 1, 'single'));
        headW2 = dlarray(0.01 * randn(1, cfg.dHeadH, 'single'));
        headb2 = dlarray(zeros(1, 1, 'single'));

        % Adam optimizer states.
        avgG_enc = [];
        avgSqG_enc = [];
        avgG_head = cell(8, 1);
        avgSqG_head = cell(8, 1);

        iter = 0;

        % Training loop.
        for ep = 1:cfg.maxEpochs

            order = randperm(numel(tr));

            for ii = 1:numel(order)

                iter = iter + 1;
                idx_in_tr = order(ii);
                p = tr(idx_in_tr);

                raw_pack = X_img(:, :, 1, p);
                dlX = dlarray(reshape(raw_pack', [1 cfg.M 1 cfg.nCell]), 'SSCB');
                y_true = dlarray(y_train_norm(idx_in_tr));

                % Compute gradients.
                [~, grads] = dlfeval(@model_grads, dlX, y_true, currEnc, enc_ref, ...
                    attW1, attb1, attW2, attb2, ...
                    headW1, headb1, headW2, headb2, cfg, current_method);

                % Update attention module and regression head.
                headParams = {attW1, attb1, attW2, attb2, ...
                    headW1, headb1, headW2, headb2};

                headGrads = {grads.attW1, grads.attb1, grads.attW2, grads.attb2, ...
                    grads.headW1, grads.headb1, grads.headW2, grads.headb2};

                if cfg.clip_head > 0
                    headGrads = local_clip_grads_cell(headGrads, cfg.clip_head);
                end

                if isempty(avgG_head{1})
                    for k = 1:8
                        avgG_head{k} = [];
                        avgSqG_head{k} = [];
                    end
                end

                for k = 1:8
                    [headParams{k}, avgG_head{k}, avgSqG_head{k}] = adamupdate( ...
                        headParams{k}, headGrads{k}, avgG_head{k}, avgSqG_head{k}, ...
                        iter, cfg.lr_head, cfg.beta1, cfg.beta2);
                end

                [attW1, attb1, attW2, attb2, ...
                    headW1, headb1, headW2, headb2] = headParams{:};

                % Update the encoder after the warm-up stage.
                if ep > cfg.warmupEpochs

                    genc = grads.enc;
                    genc = local_mask_enc_grads_keep_last_conv(currEnc, genc, cfg.tune_last_conv);

                    if cfg.clip_enc > 0
                        genc = local_clip_grads_table(genc, cfg.clip_enc);
                    end

                    [currEnc, avgG_enc, avgSqG_enc] = adamupdate( ...
                        currEnc, genc, avgG_enc, avgSqG_enc, ...
                        iter, cfg.lr_enc, cfg.beta1, cfg.beta2);
                end
            end
        end

        % Inference on the held-out pack.
        raw_te = X_img(:, :, 1, te);
        dlX_te = dlarray(reshape(raw_te', [1 cfg.M 1 cfg.nCell]), 'SSCB');

        dlZ = predict(currEnc, dlX_te);
        Z_te = stripdims(squeeze(dlZ));

        [ypred_norm, w_te] = model_predict_head_aligned( ...
            Z_te, attW1, attb1, attW2, attb2, ...
            headW1, headb1, headW2, headb2, cfg, current_method);

        w_num = double(gather(extractdata(w_te)));
        Cell_Attn_All(:, te) = w_num(:);
        MaxAttn_List(te) = max(w_num);

        val_norm = gather(extractdata(ypred_norm));
        yhat(te) = val_norm * sgY + muY;

        fprintf('  [Fold %d] True=%.3f | Pred=%.3f\n', ...
            te, y_all(te), yhat(te));
    end

    % Evaluate the current pooling mechanism.
    err = yhat - y_all;

    RMSE = sqrt(mean(err.^2));
    MAE  = mean(abs(err));
    MAPE = mean(abs(err ./ y_all)) * 100;

    sst = sum((y_all - mean(y_all)).^2);
    sse = sum(err.^2);
    R2  = 1 - sse / sst;

    fprintf('[%s] Final result: R2 = %.4f | RMSE = %.4f | MAE = %.4f | MAPE = %.2f%%\n', ...
        current_method, R2, RMSE, MAE, MAPE);

    Results.(current_method).yhat = yhat;
    Results.(current_method).err = err;
    Results.(current_method).MaxAttn_List = MaxAttn_List;
    Results.(current_method).Cell_Attn_All = Cell_Attn_All;
    Results.(current_method).RMSE = RMSE;
    Results.(current_method).MAE = MAE;
    Results.(current_method).MAPE = MAPE;
    Results.(current_method).R2 = R2;
end

%% 4) Export prediction results and attention matrices
fprintf('\n======================================================\n');
fprintf('Exporting prediction results and attention matrices...\n');

All_Metrics_Table = table();

for m = 1:num_methods

    m_name = pool_methods{m};

    % Pack-level SOH prediction results.
    Export_SOH = table( ...
        (1:Npack)', ...
        SN_List, ...
        double(y_all(:)), ...
        double(Results.(m_name).yhat(:)), ...
        double(Results.(m_name).err(:)), ...
        abs(double(Results.(m_name).err(:))), ...
        double(Results.(m_name).MaxAttn_List(:)), ...
        'VariableNames', { ...
        'Pack_Index', ...
        'Pack_ID', ...
        'True_SOH', ...
        'Predicted_SOH', ...
        'Error', ...
        'Abs_Error', ...
        'Max_Attention_Score'});

    % Cell-level attention-score matrix.
    cell_col_names = arrayfun(@(x) sprintf('Pack_%d', x), ...
        1:Npack, 'UniformOutput', false);

    Export_Cell_Attn = array2table( ...
        Results.(m_name).Cell_Attn_All, ...
        'VariableNames', cell_col_names);

    Export_Cell_Attn = [ ...
        table((1:cfg.nCell)', 'VariableNames', {'Cell_Index'}), ...
        Export_Cell_Attn];

    % Method-level performance metrics.
    Export_Metrics = table( ...
        {m_name}, ...
        double(Results.(m_name).R2), ...
        double(Results.(m_name).RMSE), ...
        double(Results.(m_name).MAE), ...
        double(Results.(m_name).MAPE), ...
        'VariableNames', { ...
        'Model_Type', ...
        'R2', ...
        'RMSE', ...
        'MAE', ...
        'MAPE'});

    All_Metrics_Table = [All_Metrics_Table; Export_Metrics];

    % Save method-specific files.
    pred_csv = fullfile(output_results_dir, sprintf('P2_Exp07_AttentionPoolingAblation_%s_PackPredictions.csv', m_name));
    attn_csv = fullfile(output_results_dir, sprintf('P2_Exp07_AttentionPoolingAblation_%s_CellAttention.csv', m_name));

    writetable(Export_SOH, pred_csv);
    writetable(Export_Cell_Attn, attn_csv);

    Results.(m_name).Export_SOH = Export_SOH;
    Results.(m_name).Export_Cell_Attn = Export_Cell_Attn;

    fprintf('  - Saved pack predictions: %s\n', pred_csv);
    fprintf('  - Saved cell attention:   %s\n', attn_csv);
end

% Save the overall metric summary table.
writetable(All_Metrics_Table, cfg.metrics_csv);

% Save complete MATLAB result file for downstream analysis.
save(cfg.result_mat, ...
    'cfg', ...
    'cfgT', ...
    'SN_List', ...
    'y_all', ...
    'pool_methods', ...
    'Results', ...
    'All_Metrics_Table');

fprintf('  - Saved overall metric summary: %s\n', cfg.metrics_csv);
fprintf('  - Saved MATLAB result file:     %s\n', cfg.result_mat);
fprintf('All files exported successfully.\n');

fprintf('\n========== P2_Exp07 attention-pooling ablation completed ==========\n');

%% Local helper functions
function [loss, grads] = model_grads(X, y, net, net_ref, ...
    attW1, attb1, attW2, attb2, ...
    headW1, headb1, headW2, headb2, cfg, pool_method)

    [Z_raw, ~] = predict(net, X);
    Z = stripdims(squeeze(Z_raw));

    [ypred, ~] = model_predict_head_aligned( ...
        Z, attW1, attb1, attW2, attb2, ...
        headW1, headb1, headW2, headb2, cfg, pool_method);

    mse_loss = mean((ypred - y).^2);

    l2_head_term = sum(attW1(:).^2) + ...
                   sum(attW2(:).^2) + ...
                   sum(headW1(:).^2) + ...
                   sum(headW2(:).^2);

    l2_sp_term = 0;

    if cfg.lambda_sp > 0

        T_curr = net.Learnables;
        T_ref  = net_ref.Learnables;

        for i = 1:height(T_curr)
            l2_sp_term = l2_sp_term + sum((T_curr.Value{i} - T_ref.Value{i}).^2, 'all');
        end
    end

    loss = mse_loss + cfg.l2_head * l2_head_term + cfg.lambda_sp * l2_sp_term;

    all_grads = dlgradient(loss, ...
        {net.Learnables, attW1, attb1, attW2, attb2, ...
        headW1, headb1, headW2, headb2});

    grads.enc    = all_grads{1};
    grads.attW1  = all_grads{2};
    grads.attb1  = all_grads{3};
    grads.attW2  = all_grads{4};
    grads.attb2  = all_grads{5};
    grads.headW1 = all_grads{6};
    grads.headb1 = all_grads{7};
    grads.headW2 = all_grads{8};
    grads.headb2 = all_grads{9};
end

function [ypred, w] = model_predict_head_aligned(Z, ...
    attW1, attb1, attW2, attb2, ...
    headW1, headb1, headW2, headb2, cfg, pool_method)

    nCell = size(Z, 2);

    switch pool_method

        case 'Mean'
            w = dlarray(ones(1, nCell, 'single') / nCell);
            zbag = Z * w.';

        case 'Standard'
            H = relu(attW1 * Z + attb1);
            score = attW2 * H + attb2;
            score = score - max(score, [], 2);
            w = exp(score) ./ sum(exp(score), 2);
            zbag = Z * w.';

        case 'Softmin'
            H = relu(attW1 * Z + attb1);
            score = -(attW2 * H + attb2) / cfg.attn_tau;
            score = score - max(score, [], 2);
            w = exp(score) ./ sum(exp(score), 2);
            zbag = Z * w.';

        case 'Hardmin'
            H = relu(attW1 * Z + attb1);
            hard_tau = 0.001;
            score = -(attW2 * H + attb2) / hard_tau;
            score = score - max(score, [], 2);
            w = exp(score) ./ sum(exp(score), 2);
            zbag = Z * w.';

        otherwise
            error('Unknown pooling method: %s', pool_method);
    end

    h1 = relu(headW1 * zbag + headb1);
    ypred = headW2 * h1 + headb2;
end

function genc = local_mask_enc_grads_keep_last_conv(net, genc, keep_last_conv)

    L = net.Layers;
    layerNames = string({L.Name});
    isConv = false(size(L));

    for i = 1:numel(L)

        cls = class(L(i));

        if contains(cls, "Convolution", "IgnoreCase", true) || ...
                contains(L(i).Name, "conv", "IgnoreCase", true)
            isConv(i) = true;
        end
    end

    convIdx = find(isConv);

    if isempty(convIdx)
        uniq = unique(string(genc.Layer), 'stable');
        keep = uniq(max(1, end-4):end);
    else
        k = min(keep_last_conv, numel(convIdx));
        keep = layerNames(convIdx(end-k+1:end));
    end

    for i = 1:height(genc)

        if ~ismember(string(genc.Layer(i)), keep)

            gi = genc.Value{i};

            if isempty(gi)
                continue;
            end

            gi_data = extractdata(gi);
            genc.Value{i} = dlarray(zeros(size(gi_data), 'like', gi_data));
        end
    end
end

function gtab = local_clip_grads_table(gtab, clipnorm)

    ss = 0;

    for i = 1:height(gtab)

        gi = gtab.Value{i};

        if ~isempty(gi)
            ss = ss + sum(extractdata(gi).^2, 'all');
        end
    end

    gn = sqrt(ss + 1e-12);

    if gn > clipnorm

        scale = clipnorm / gn;

        for i = 1:height(gtab)

            if ~isempty(gtab.Value{i})
                gtab.Value{i} = gtab.Value{i} * scale;
            end
        end
    end
end

function gcell = local_clip_grads_cell(gcell, clipnorm)

    ss = 0;

    for i = 1:numel(gcell)

        gi = gcell{i};

        if ~isempty(gi)
            ss = ss + sum(extractdata(gi).^2, 'all');
        end
    end

    gn = sqrt(ss + 1e-12);

    if gn > clipnorm

        scale = clipnorm / gn;

        for i = 1:numel(gcell)

            if ~isempty(gcell{i})
                gcell{i} = gcell{i} * scale;
            end
        end
    end
end