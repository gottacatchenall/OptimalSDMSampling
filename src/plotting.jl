function get_rangediff(begin_range::SDMLayer, end_range::SDMLayer)
    change_map = Int32.(deepcopy(begin_range))
    change_map.grid .= 0

    loss = begin_range .& .!end_range
    same = begin_range .& end_range
    gain = .!begin_range .& end_range

    change_map.grid[findall(gain.grid)] .= 3
    change_map.grid[findall(same.grid)] .= 2
    change_map.grid[findall(loss.grid)] .= 1
    return change_map
end 

function plot_rangediff(begin_range::SDMLayer, end_range::SDMLayer)
    change_map = get_rangediff(begin_range, end_range)
    heatmap(change_map, colormap=[:grey90,"#e17878ff",:grey40,"#78aee1ff" ])
end

function plot_rangediff(sr::ShiftingRange, start_t=1, end_t=8)

    begin_range = sr.scores[start_t] .> sr.threshold
    end_range = sr.scores[end_t] .> sr.threshold

    plot_rangediff(begin_range, end_range)

end 


function plot_rangeshift(sr::ShiftingRange; size=(1000, 1000))
    K,T = num_predictors(sr), num_timesteps(sr)

    f = Figure(size=size)
    for i in CartesianIndices((1:K+1, 1:T))
        row = i[1]
        col = i[2]
        ylabel = col == 1 ?  (row == 1 ? "Range" : "Layer $(row-1)") : ""
        title = row == 1 ? "T = $col" : ""

        ax = Axis(
            f[i[1],i[2]], 
            aspect=1, 
            title=title, 
            ylabel=ylabel,
            xticksvisible=false, 
            yticksvisible=false, 
            xticklabelsvisible=false, 
            yticklabelsvisible=false
        )
        mat = row == 1 ? sr.scores[col] .> sr.threshold : sr.predictors[col][row-1]
        
        crange = (0,1) # row == 1 ? (0,1) : extrema(sr.predictors[:,:,row-1, :])
        heatmap!(ax, mat, colorrange=crange)
    end

    for k in 2:K+1
        X = 0:0.01:1
        xlab = k == K+1 ? "Value" : ""
        ax = Axis(f[k,T+1], aspect=1, xticks=0:1, yticks=0:1, xlabel=xlab, ylabel="Suitability")
        limits!(ax, 0, 1, 0, 1)
        lines!(ax, X, logistic(sr.niche.shapes[k-1], sr.niche.centers[k-1]).(X))


    end
    f
end 