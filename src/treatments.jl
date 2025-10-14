
function adaptive_bon(n_obs, domain, u, α)
    BONs.sample(
        BalancedAcceptance(n_obs), 
        domain, 
        inclusion = BONs.tilt(u, α)
    )
end

function spatially_balanced_bon(n_obs, domain, u, α)
    BONs.sample(
        BalancedAcceptance(n_obs), 
        domain, 
    )
end


function random_bon(n_obs, domain, u, α)
    BONs.sample(
        SimpleRandom(n_obs), 
        domain, 
    )
end


function adaptive_sampling_treament(
    sr::ShiftingRange;
    total_samples = 100,
    baseline_proportion = 0.2,
    baseline_method = BalancedAcceptance,
    update_method = adaptive_bon,
    α = 5.
)
    domain = sr.predictors[1][1]

    n_t = num_timesteps(sr)
    n_baseline = Int(ceil(baseline_proportion*total_samples))
    obs_per_future_timestep = split_observations(total_samples - n_baseline, n_t - 1)    

    baseline_bon = BONs.sample(baseline_method(n_baseline), domain)
    baseline_occ = sample_occurrence(sr, baseline_bon; t = 1)

    X, y, ts = get_features_and_labels(sr, baseline_occ)

    model, τ = fit_sdm(Matrix(X'),y)
    r,u = predict_sdm(model, sr.predictors[ts[begin]], τ)
    rescale!(u)

    ranges, uncertainties = [r], [u]

    for t in 2:n_t 
        bon_t = update_method(obs_per_future_timestep[t-1], domain, u, α) 

        occ_t = sample_occurrence(sr, bon_t; t = t)

        Xₜ, yₜ, tₜ = get_features_and_labels(sr, occ_t)

        X = hcat(X, Xₜ)
        y = vcat(y, yₜ)
        ts = vcat(ts, tₜ)

        model, τ = fit_sdm(Matrix(X'),y)
        r,u = predict_sdm(model, sr.predictors[t], τ)

        rescale!(u)
        push!(ranges, r)
        push!(uncertainties, u)
    end 
    ranges, uncertainties
end

function get_ranges(sr)
    return [sr.scores[t] .> sr.threshold for t in 1:num_timesteps(sr)]
end



function run_treatment(
    sr;
    total_samples = 150,
    baseline_proportion = 0.25,
    tilting = 3.,
    method = adaptive_bon
)
    true_ranges = [sr.scores[t] .> sr.threshold for t in 1:num_timesteps(sr)]
    predicted_ranges, _ = adaptive_sampling_treament(
        sr; 
        baseline_proportion = baseline_proportion, 
        total_samples = total_samples, 
        α = tilting,
        update_method = method
    )
    DataFrame(compute_metrics(true_ranges, predicted_ranges))
end 
