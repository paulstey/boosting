#!/bin/bash

#SBATCH --time=4:30:00
#SBATCH -c 12
#SBATCH --mem 16g
#SBATCH --job-name=n200minclass10


module load julia/0.5.0
julia -p 8 sim.jl 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
