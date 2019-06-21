#!/bin/bash

#SBATCH --time=5:59:00
#SBATCH -c 4
#SBATCH --mem 32g
#SBATCH --job-name=minclass10



module load julia/0.5.0
julia -p 3 sim.jl
