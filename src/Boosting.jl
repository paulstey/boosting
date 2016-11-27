using Compat

import length 

if VERSION >= v"0.4.0-dev"
    typealias Range1{Int} Range{Int}
    _int(x) = round(Integer, x)
    float(x) = map(Float64, x)
else
    _int(x) = int(x)
end
squeeze(v::Vector, i::Integer) = v

const NO_BEST=(0,0)

immutable Leaf
    majority::Any
    values::Vector
end

@compat immutable Node
    featid::Integer
    featval::Any
    left::Union{Leaf,Node}
    right::Union{Leaf,Node}
end

@compat typealias LeafOrNode Union{Leaf,Node}

immutable Ensemble
    trees::Vector{Node}
end

length(ensemble::Ensemble) = length(ensemble.trees)

include("measures.jl")
include("classification.jl")
