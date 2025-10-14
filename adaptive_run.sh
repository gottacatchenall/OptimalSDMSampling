#!/bin/bash
#SBATCH --account=def-tpoisot
#SBATCH --job-name=AdaptiveRangeShiftSampling 
#SBATCH --output=%x-%A-%a.out
#SBATCH --nodes=1               
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=16G 
#SBATCH --array=1-512 
#SBATCH --time=1:00:00         

export JULIA_DEPOT_PATH="/project/def-tpoisot/mcatchen/JuliaEnvironments/OptimalSDMSampling"

module load julia/1.11.3
julia main.jl adaptive