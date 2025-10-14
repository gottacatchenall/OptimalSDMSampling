struct ShiftingRange
    predictors
    scores
    threshold
    niche
end
Base.show(io::IO, sr::ShiftingRange) = print(io, "ShiftingRange with $(num_predictors(sr)) predictors and $(num_timesteps(sr)) timesteps")

num_predictors(sr::ShiftingRange) = length(sr.predictors[1])
num_timesteps(sr::ShiftingRange) = length(sr.predictors)

function predict_range(niche, L)
    score = deepcopy(L[begin])
    for i in eachindex(L[begin])
        score[i] = predict(niche, [L[j][i] for j in eachindex(L)])
    end
    return score
end

function predict_future_range(niche, L, threshold)    
    score = predict_range(niche, L)
    score
end

function predict_current_range(niche, L; prevalence = 0.25)
    score = predict_range(niche, L)
    threshold = quantile(score, 1 - prevalence)
    return score, threshold
end


function get_shifting_range(niche, bioclim)
    current_score, τ = predict_current_range(niche, bioclim[begin])
    futures = [predict_future_range(niche, bc, τ) for bc in bioclim[2:end]]
    scores = vcat(current_score, futures)
    
    ShiftingRange(bioclim, scores, τ, niche)
end