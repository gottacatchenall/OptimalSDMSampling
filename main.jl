using SpeciesDistributionToolkit
using CairoMakie
using GeoMakie
using MultivariateStats
using BiodiversityObservationNetworks
using Statistics
using Distributions
using Random
using XGBoost
using EvoTrees
using LinearAlgebra
using BenchmarkTools
using DataFrames, CSV
using Logging
using ProgressMeter

const BONs = BiodiversityObservationNetworks
const MV = MultivariateStats
const SDT = SpeciesDistributionToolkit

include("bioclim.jl")
include("niche.jl")
include("rangeshift.jl")
include("occurrence.jl")
include("fnl.jl")
include("plotting.jl")


function get_features_and_labels(sr::ShiftingRange, occ::Occurrences)
    t = occ.time
    feat = Matrix(hcat([[predictor[n...] for n in occ.bon] for predictor in sr.predictors[t]]...)')
    return feat, occ.labels, [t for _ in eachindex(occ.labels)]
end

function predict_sdm(models, predictors, τ)
    Xp = Float32.([predictors[i][k] for k in keys(predictors[1]), i in eachindex(predictors)])

    preds = hcat([XGBoost.predict(m, DMatrix(Xp)) for m in models]...)

    prediction = similar(first(predictors), Float32)
    uncertainty = similar(first(predictors), Float32)

    prediction.grid[prediction.indices] .= mean(preds, dims=2)
    uncertainty.grid[prediction.indices] .= var(preds, dims=2)

    return prediction .> τ, uncertainty
end

function optimal_threshold(labels, predictions)
    τs = 0f0:0.01f0:1f0
    τs[findmax([mcc(ConfusionMatrix(predictions, labels, τ)) for τ in τs])[2]]
end 

function fit_sdm(features, labels, n_bags= 30, n_rounds = 50)
    # Train models on bootstraps
    n = length(labels)
    models = XGBoost.Booster[]
    for i in 1:n_bags
        idxs = rand(1:n, n)  # bootstrap resample
        dtrain = DMatrix(features[idxs, :], label = labels[idxs])
        booster = xgboost(dtrain, num_round = n_rounds, verbosity=0)        
        push!(models, booster)
    end

    preds = vec(mean(hcat([XGBoost.predict(m, features) for m in models]...), dims=2))

    
    τ = optimal_threshold(labels, preds)
    return models, τ
end

function split_observations(N, t)
    idx_to_add = randperm(t)[begin:(N % t)]
    fill(div(N, t), t) .+ map(x->x∈idx_to_add, 1:t)
end

function adaptive_sampling_treament(
    sr::ShiftingRange;
    total_samples = 100,
    baseline_proportion = 0.2,
    baseline_method = BalancedAcceptance,
    α = 3.
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
        bon_t = BONs.sample(
            BalancedAcceptance(obs_per_future_timestep[t-1]), 
            domain, 
            inclusion = BONs.tilt(u, α)
        )
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

function compute_errors(true_range::SDMLayer, predicted_range::SDMLayer)    
    underestim = true_range .& .!predicted_range
    overestim = .!true_range .& predicted_range
    total_size = length(findall(true_range.indices))
    over_area = length(findall(overestim)) / total_size
    under_area = length(findall(underestim)) / total_size
    return under_area, over_area
end

function compute_loss_regions(sr)
    ranges = get_ranges(sr)
    lost = Int.(deepcopy(ranges[begin]))
    lost.grid .= 0
    for t in 2:num_timesteps(sr)
        lost.grid[findall(ranges[t-1] .& .!ranges[t])] .= t
    end
    return lost
end

function compute_gain_regions(sr)
    ranges = get_ranges(sr)
    gain = Int.(deepcopy(ranges[begin]))
    gain.grid .= 0
    for t in 2:num_timesteps(sr)
        gain.grid[findall(.!ranges[t-1] .& ranges[t])] .= t
    end
    return gain
end

total_area(x) = length(findall(x.indices))
area(binary_layer) = length(findall(binary_layer))
area_lost(prev_range, future_range) = area(prev_range .& .!future_range)/total_area(!future_range)
area_gained(prev_range, future_range) = area(.!prev_range .& future_range)/total_area(!future_range)
range_area(range) = area(range)/total_area(range)
range_error(true_range, predicted_range) = abs(range_area(true_range) - range_area(predicted_range))
function captured_loss(predicted_prev, predicted_future, true_prev, true_future)
    true_loss = true_prev & .!true_future
    predicted_loss = predicted_prev & .!predicted_future
    # What proportion of lost range was actually captured by prediction? 
    area(true_loss .& predicted_loss) / area(true_loss)
end
function captured_gain(predicted_prev, predicted_future, true_prev, true_future)
    true_gain = .!true_prev & true_future
    predicted_gain = .!predicted_prev & predicted_future
    # What proportion of gained range was actually captured by prediction? 
    area(true_gain .& predicted_gain) / area(true_gain)
end

function captured_diff(predicted_prev, predicted_future, true_prev, true_future)
    true_gain = .!true_prev & true_future
    predicted_gain = .!predicted_prev & predicted_future
    true_loss = true_prev & .!true_future
    predicted_loss = predicted_prev & .!predicted_future

    # What proportion of gained range was actually captured by prediction? 
    (area(true_loss .& predicted_loss)  + area(true_gain .& predicted_gain)) / (area(true_loss)+area(true_gain))
end

function compute_metrics(
    true_ranges,
    predicted_ranges
)

    dicts = Dict[]

    for t in 2:length(true_ranges)
        dict = Dict{Symbol,Real}()

        dict[:timestep] = t
        dict[:range_area] = range_area(true_ranges[t])
        dict[:true_area_loss] = area_lost(true_ranges[t-1], true_ranges[t])
        dict[:true_area_gain] = area_gained(true_ranges[t-1], true_ranges[t])

        dict[:predicted_area_loss] = area_lost(predicted_ranges[t-1], predicted_ranges[t])
        dict[:predicted_area_gain] = area_gained(predicted_ranges[t-1], predicted_ranges[t])

        dict[:captured_loss] = captured_loss(predicted_ranges[t-1], predicted_ranges[t],true_ranges[t-1], true_ranges[t])
        dict[:captured_gain] = captured_gain(predicted_ranges[t-1], predicted_ranges[t],true_ranges[t-1], true_ranges[t])
        dict[:captured_change] = captured_diff(predicted_ranges[t-1], predicted_ranges[t],true_ranges[t-1], true_ranges[t])
        push!(dicts, dict)
    end 

    dicts
end



function run_treatment(
    bioclim,
    sr;
    total_samples = 150,
    baseline_proportion = 0.25,
    tilting = 3.
)
    true_ranges = [sr.scores[t] .> sr.threshold for t in 1:num_timesteps(sr)]
    predicted_ranges, _ = adaptive_sampling_treament(
        sr; 
        baseline_proportion = baseline_proportion, 
        total_samples = total_samples, 
        α = tilting
    )


    DataFrame(compute_metrics(true_ranges, predicted_ranges))
end 

#ssp_dir = joinpath("data", "CHELSA_rescaled", "ssp245")
#bioclim = read_bioclim(ssp_dir)
#df = run_treatment(RangeShiftTreatment(), bioclim)
# TODO: the BON has a chance to brick if the uncertainty surface is so uneven 
# that the odds any particular value is below the 3rd halton dimension is extremely low

function main()
    global_logger(ConsoleLogger(Error));

    bioclim = read_bioclim(joinpath("data", "CHELSA_rescaled", "CHELSA_rescaled_shorter"))


    prop_baseline = 0.25:0.05:0.75
    Ntotals = 50:50:500
    tilting = 3.

    dfs = []

    prog = Progress(length(Ntotals)*length(prop_baseline))

    num_layers = length(bioclim[begin])
    niche = LogisticNiche(
        centers = rand(Uniform(0.1, 0.5), num_layers),
        shapes = rand(Normal(0., 0.3), num_layers)
    )
    sr = get_shifting_range(niche, bioclim)

    for p in prop_baseline
        for n in Ntotals
            df = run_treatment(
                bioclim,
                sr;
                total_samples = n,
                baseline_proportion = p,
                tilting = tilting
            )
            df.n_total = fill(n, nrow(df))
            df.prop_baseline = fill(p, nrow(df))
            push!(dfs, df)
            next!(prog)
        end 
    end 
    total_df = vcat(dfs...)
    return total_df
end 

main()