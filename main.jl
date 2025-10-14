using SpeciesDistributionToolkit
using CairoMakie
using GeoMakie
using MultivariateStats
using BiodiversityObservationNetworks
using Statistics
using Distributions
using Random
using XGBoost
using LinearAlgebra
using DataFrames, CSV
using Logging
using ProgressMeter

const BONs = BiodiversityObservationNetworks
const MV = MultivariateStats
const SDT = SpeciesDistributionToolkit


include(joinpath("src", "bioclim.jl"))
include(joinpath("src", "niche.jl"))
include(joinpath("src", "rangeshift.jl"))
include(joinpath("src", "occurrence.jl"))
include(joinpath("src", "fnl.jl"))
include(joinpath("src", "plotting.jl"))
include(joinpath("src", "sdm.jl"))
include(joinpath("src", "metrics.jl"))
include(joinpath("src", "treatments.jl"))



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