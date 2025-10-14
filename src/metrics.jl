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
        dict[:range_error] = range_error(true_ranges[t], predicted_ranges[t])

        push!(dicts, dict)
    end 

    dicts
end

