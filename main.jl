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
# using ProgressMeter

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

    method_name = ARGS[1]

    method = spatially_balanced_bon
    if method_name == "adaptive"
        method = adaptive_bon
    elseif method_name == "random"
        method = random_bon
    elseif method_name == "balanced"
        method = spatially_balanced_bon
    end

    bioclim = read_bioclim(joinpath("data"))

    prop_baseline = 0.25:0.05:0.75
    Ntotals = 50:50:500
    tilting = 5.

    dfs = []

    #prog = Progress(length(Ntotals)*length(prop_baseline))

    num_layers = length(bioclim[begin])
    niche = LogisticNiche(
        centers = rand(Uniform(0.1, 0.5), num_layers),
        shapes = rand(Normal(0., 0.3), num_layers)
    )
    sr = get_shifting_range(niche, bioclim)

    for p in prop_baseline
        for n in Ntotals
            df = run_treatment(
                sr;
                total_samples = n,
                baseline_proportion = p,
                tilting = tilting,
                method = method
            )
            df.n_total = fill(n, nrow(df))
            df.prop_baseline = fill(p, nrow(df))
            push!(dfs, df)
            #next!(prog)
        end 
    end 
    total_df = vcat(dfs...)


    outdir = joinpath("artifacts", method_name)
    mkpath(outdir)

    job_id = ENV["SLURM_ARRAY_TASK_ID"]
    CSV.write(joinpath("artifacts", outdir, "replicate_$job_id.csv"), total_df)

    return total_df
end 



main()