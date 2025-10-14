

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