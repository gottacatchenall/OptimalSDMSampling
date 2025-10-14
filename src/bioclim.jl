function read_bioclim(ssp_dir, yr)
    yr_dir = joinpath(ssp_dir, yr)

    bioclim_filenames = readdir(yr_dir)
    perm = sortperm([parse(Int,split(split(b, ".tif")[begin], "bio")[end]) for b in bioclim_filenames])
    
    [SDMLayer(joinpath(yr_dir, l)) for l in bioclim_filenames[perm]]
end

function sort_year_dirs(dir_names) 
    filter!(!isequal(".DS_Store"), dir_names)  
    perm = sortperm([parse(Int, split(d, "-")[begin]) for d in dir_names])
    return dir_names[perm]
end

function read_bioclim(ssp_dir)
    year_dirs = sort_year_dirs(readdir(ssp_dir))
    raw_baseline = read_bioclim(ssp_dir, year_dirs[begin])
    pca = MV.fit(MV.PCA, raw_baseline, maxoutdim=3, pratio=0.999)
    pca_layers = [MV.predict(pca, read_bioclim(ssp_dir, yr)) for yr in year_dirs]

    # rescale each layer based on extrema 
    for l_i in eachindex(pca_layers[begin])
        layer_max = -Inf
        layer_min = Inf
        for y_i in eachindex(year_dirs)
            m, M = extrema(pca_layers[y_i][l_i])

            layer_min = min(m, layer_min)
            layer_max = max(M, layer_max)
        end

        for y_i in eachindex(year_dirs)
            pca_layers[y_i][l_i] = (pca_layers[y_i][l_i] .- layer_min) ./ (layer_max-layer_min)
        end
    end
    return pca_layers
end 

