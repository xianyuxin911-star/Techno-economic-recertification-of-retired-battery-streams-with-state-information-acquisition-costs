function [sampled_noise, diagnostics] = local_build_noise_by_k( ...
    k_current, ...
    k_emp, k_anchor, k_bridge_end, ...
    err_cell, ...
    w_curve, muM_curve, sigM_curve, muR_curve, sigR_curve, ...
    rmse_target, k_vals, ...
    base_u, base_z, base_pick, base_bridge)
%LOCAL_BUILD_NOISE_BY_K Generate label-dependent residual samples.
%
% Residual source:
%   k <= k_anchor:
%       empirical all-inner residual distribution at the current k;
%   k_anchor < k < k_bridge_end:
%       probability mixture of the empirical anchor distribution and the
%       k-specific parametric two-component distribution, followed by RMSE
%       calibration;
%   k >= k_bridge_end:
%       fully parametric two-component residual distribution.
%
% Inputs base_u, base_z, base_pick and base_bridge should be generated once
% outside the k loop and reused across k values. This preserves common
% random numbers while keeping the bridge selector independent of the
% empirical quantile and Gaussian draws.
%
% Optional second output diagnostics reports the source type, bridge weight
% and generated RMSE for checking consistency with Part 3 Exp03.

    %% 1) Normalize and validate inputs
    k_emp = double(k_emp(:));
    k_vals = double(k_vals(:));

    base_u = double(base_u(:));
    base_z = double(base_z(:));
    base_pick = double(base_pick(:));
    base_bridge = double(base_bridge(:));

    n_sim = numel(base_u);

    if n_sim == 0
        error('base_u must contain at least one Monte Carlo draw.');
    end

    if numel(base_z) ~= n_sim || ...
            numel(base_pick) ~= n_sim || ...
            numel(base_bridge) ~= n_sim
        error(['base_u, base_z, base_pick and base_bridge must have ' ...
            'the same number of elements.']);
    end

    if any(~isfinite(base_u)) || any(base_u < 0) || any(base_u > 1)
        error('base_u must contain finite values in [0, 1].');
    end

    if any(~isfinite(base_z))
        error('base_z must contain finite standard-normal draws.');
    end

    if any(~isfinite(base_pick)) || any(base_pick < 0) || any(base_pick > 1)
        error('base_pick must contain finite values in [0, 1].');
    end

    if any(~isfinite(base_bridge)) || ...
            any(base_bridge < 0) || any(base_bridge > 1)
        error('base_bridge must contain finite values in [0, 1].');
    end

    idx_now = find(k_vals == k_current, 1);

    if isempty(idx_now)
        error('k_current = %g is not present in k_vals.', k_current);
    end

    parameter_arrays = { ...
        w_curve, muM_curve, sigM_curve, muR_curve, sigR_curve, rmse_target};
    parameter_names = { ...
        'w_curve', 'muM_curve', 'sigM_curve', ...
        'muR_curve', 'sigR_curve', 'rmse_target'};

    for ii = 1:numel(parameter_arrays)
        if numel(parameter_arrays{ii}) ~= numel(k_vals)
            error('%s must have one value for every entry in k_vals.', ...
                parameter_names{ii});
        end
    end

    w_now = double(w_curve(idx_now));
    muM_now = double(muM_curve(idx_now));
    sigM_now = double(sigM_curve(idx_now));
    muR_now = double(muR_curve(idx_now));
    sigR_now = double(sigR_curve(idx_now));
    target_rmse = double(rmse_target(idx_now));

    if ~isfinite(w_now) || w_now < 0 || w_now > 1
        error('Invalid mixture weight at k = %g: %.6g.', k_current, w_now);
    end

    if any(~isfinite([muM_now, sigM_now, muR_now, sigR_now])) || ...
            sigM_now <= 0 || sigR_now <= 0
        error('Invalid parametric residual parameters at k = %g.', k_current);
    end

    if ~isfinite(target_rmse) || target_rmse <= 0
        error('Invalid RMSE target at k = %g.', k_current);
    end

    %% 2) Construct the k-specific parametric residual distribution
    % The latent component weight w is distinct from the fixed-region
    % probability P_R. The supplied w_curve must therefore be the selected
    % Part 3 Exp03 mixture-weight curve.
    is_right_component = base_pick < w_now;

    e_parametric = zeros(n_sim, 1);
    e_parametric(~is_right_component) = ...
        muM_now + sigM_now .* base_z(~is_right_component);
    e_parametric(is_right_component) = ...
        muR_now + sigR_now .* base_z(is_right_component);

    %% 3) Select empirical, bridged or extrapolated residual source
    diagnostics = struct();
    diagnostics.k = k_current;
    diagnostics.target_RMSE = target_rmse;
    diagnostics.bridge_alpha = 0;
    diagnostics.bridge_RMSE_scale = 1;
    diagnostics.parametric_fraction_realized = NaN;

    if k_current <= k_anchor
        % Empirical all-inner residual distribution at the current k.
        idx_emp = find(k_emp == k_current, 1);

        if isempty(idx_emp)
            error(['For k_current <= k_anchor, k_current must be present ' ...
                'in the empirical k set.']);
        end

        sampled_noise = local_empirical_quantile_sample( ...
            err_cell{idx_emp}, base_u);

        diagnostics.source_type = 'Empirical';

    elseif k_current < k_bridge_end
        % Smoothstep probability mixture between the empirical anchor and
        % the k-specific parametric distribution. Samples are selected from
        % either source; they are not averaged pointwise.
        idx_anchor = find(k_emp == k_anchor, 1);

        if isempty(idx_anchor)
            error('k_anchor = %g is not present in k_emp.', k_anchor);
        end

        e_empirical_anchor = local_empirical_quantile_sample( ...
            err_cell{idx_anchor}, base_u);

        t = (k_current - k_anchor) / (k_bridge_end - k_anchor);
        t = min(max(t, 0), 1);
        alpha = 3 * t^2 - 2 * t^3;

        use_parametric = base_bridge < alpha;

        sampled_noise = e_empirical_anchor;
        sampled_noise(use_parametric) = e_parametric(use_parametric);

        % Match the fitted RMSE trajectory in the bridge interval. This
        % calibration is intentionally not applied in the empirical or
        % fully extrapolated intervals.
        rmse_before = sqrt(mean(sampled_noise.^2));
        rmse_scale = target_rmse / max(rmse_before, 1e-12);
        sampled_noise = sampled_noise .* rmse_scale;

        diagnostics.source_type = 'Bridged';
        diagnostics.bridge_alpha = alpha;
        diagnostics.bridge_RMSE_scale = rmse_scale;
        diagnostics.parametric_fraction_realized = mean(use_parametric);

    else
        % Fully extrapolated parametric residual distribution.
        sampled_noise = e_parametric;
        diagnostics.source_type = 'Extrapolated';
        diagnostics.parametric_fraction_realized = 1;
    end

    diagnostics.generated_RMSE = sqrt(mean(sampled_noise.^2));
    diagnostics.generated_bias = mean(sampled_noise);
    diagnostics.generated_std = std(sampled_noise, 0);
end

function sampled = local_empirical_quantile_sample(empirical_values, base_u)
%LOCAL_EMPIRICAL_QUANTILE_SAMPLE Sample within the empirical residual range.
% Quantile interpolation is used to preserve the empirical distribution
% smoothly. Uniform draws are clamped to the available empirical plotting
% positions so that no artificial tail extrapolation is introduced.

    empirical_values = double(empirical_values(:));
    empirical_values = empirical_values(isfinite(empirical_values));

    if isempty(empirical_values)
        error('The empirical residual pool is empty or contains no finite values.');
    end

    empirical_values = sort(empirical_values);

    if numel(empirical_values) == 1
        sampled = repmat(empirical_values, numel(base_u), 1);
        return;
    end

    p_empirical = ((1:numel(empirical_values))' - 0.5) / ...
        numel(empirical_values);

    u_clamped = min(max(base_u, p_empirical(1)), p_empirical(end));

    sampled = interp1( ...
        p_empirical, empirical_values, u_clamped, 'linear');
end
