using SpeciesDistributionToolkit
# using CairoMakie
# using GeoMakie
using MultivariateStats
using BiodiversityObservationNetworks
using Statistics
using Distributions
using Random
using XGBoost
using LinearAlgebra
using DataFrames, CSV
using Logging

#using ProgressMeter

const BONs = BiodiversityObservationNetworks
const MV = MultivariateStats
const SDT = SpeciesDistributionToolkit


include(joinpath("src", "bioclim.jl"))
include(joinpath("src", "niche.jl"))
include(joinpath("src", "rangeshift.jl"))
include(joinpath("src", "occurrence.jl"))
include(joinpath("src", "fnl.jl"))
#include(joinpath("src", "plotting.jl"))
include(joinpath("src", "sdm.jl"))
include(joinpath("src", "metrics.jl"))
include(joinpath("src", "treatments.jl"))

function main()
    global_logger(ConsoleLogger(Error));

    method = adaptive_bon
   

    bioclim = read_bioclim(joinpath("data"))

    n = 250
    alphas = 1:0.5:10

    dfs = []

    #prog = Progress(length(alphas))

    num_layers = length(bioclim[begin])
    niche = LogisticNiche(
        centers = rand(Uniform(0.1, 0.5), num_layers),
        shapes = rand(Normal(0., 0.3), num_layers)
    )
    sr = get_shifting_range(niche, bioclim)

    for tilting in alphas
        df = run_treatment(
            sr;
            total_samples = n,
            tilting = tilting,
            method = method
        )
        df.n_total = fill(n, nrow(df))
        df.alpha = fill(tilting, nrow(df))
        push!(dfs, df)
        #next!(prog)
    end 
    total_df = vcat(dfs...)


    outdir = joinpath("artifacts", "vary_alpha")
    mkpath(outdir)

    job_id = ENV["SLURM_ARRAY_TASK_ID"]
    CSV.write(joinpath(outdir, "replicate_$job_id.csv"), total_df)

    return total_df
end 



main()