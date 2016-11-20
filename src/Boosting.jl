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