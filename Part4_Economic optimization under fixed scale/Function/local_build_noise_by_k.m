function sampled_noise = local_build_noise_by_k( ...
    k_current, ...
    k_emp, k_anchor, k_bridge_end, ...
    err_cell, ...
    w_curve, muM_curve, sigM_curve, muR_curve, sigR_curve, rmse_target, k_vals, ...
    base_u, base_z, base_pick)

    idx_now = find(k_vals == k_current, 1);
    if isempty(idx_now)
        error('k_current 不在 k_vals 中。');
    end

    % 参数双模态分布
    isR = base_pick < w_curve(idx_now);
    e_par = zeros(size(base_u));
    e_par(~isR) = muM_curve(idx_now) + sigM_curve(idx_now) .* base_z(~isR);
    e_par(isR)  = muR_curve(idx_now) + sigR_curve(idx_now) .* base_z(isR);

    % 经验锚点
    idx_anchor = find(k_emp == k_anchor, 1);
    err_anchor = err_cell{idx_anchor};

    if k_current <= k_anchor
        % 经验 bootstrap
        idx_emp = find(k_emp == k_current, 1);
        if isempty(idx_emp)
            error('k_current <= k_anchor 时，必须在经验 k 集合中。');
        end
        e_data = sort(err_cell{idx_emp});
        p = ((1:numel(e_data))' - 0.5) / numel(e_data);
        sampled_noise = interp1(p, e_data, base_u, 'linear', 'extrap');

    elseif k_current < k_bridge_end
        % 桥接区间
        t = (k_current - k_anchor) / (k_bridge_end - k_anchor);
        alpha = 3*(min(max(t,0),1))^2 - 2*(min(max(t,0),1))^3;

        e_data = sort(err_anchor);
        p = ((1:numel(e_data))' - 0.5) / numel(e_data);
        e_emp = interp1(p, e_data, base_u, 'linear', 'extrap');

        sampled_noise = (1 - alpha) .* e_emp + alpha .* e_par;
        sampled_noise = sampled_noise .* ...
            (rmse_target(idx_now) / max(sqrt(mean(sampled_noise.^2)), 1e-12));

    else
        % 完全参数化外推
        sampled_noise = e_par;
    end
end