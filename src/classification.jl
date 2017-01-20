# Utilities



# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
label_index(labels) = Dict([Pair(v => k) for (k, v) in enumerate(labels)])

## Helper function. Counts the votes.
## Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
## order as get_labels(classifier) (eg. ["versicolor", "setosa", "virginica"])
function compute_probabilities(labels::Vector, votes::Vector, weights=1.0)
    label2ind = label_index(labels)
    counts = zeros(Float64, length(label2ind))

    for (i, label) in enumerate(votes)
        if isa(weights, Number)
            counts[label2ind[label]] += weights
        else
            counts[label2ind[label]] += weights[i]
        end
    end
    soln = counts / sum(counts) # normalize to get probabilities

    if any(map(x -> isnan(x), soln))
        print(counts)
        warn("Found NaN")
    end
    return soln
end

# Applies `row_fun(X_row)::Vector` to each row in X
# and returns a Matrix containing the resulting vectors, stacked vertically
function stack_function_results(row_fun::Function, X::Matrix)
    N = size(X, 1)
    N_cols = length(row_fun(squeeze(X[1,:],1))) # gets the number of columns
    out = Array(Float64, N, N_cols)
    for i in 1:N
        out[i, :] = row_fun(squeeze(X[i,:],1))
    end
    return out
end


function _split(labels::Vector, features::Matrix, nsubfeatures::Int, weights::Vector, rng::AbstractRNG)
    if weights == [0]
        _split_info_gain(labels, features, nsubfeatures, rng)
    else
        _split_neg_z1_loss(labels, features, weights)
    end
end

function _split_info_gain(labels::Vector, features::Matrix, nsubfeatures::Int,
                          rng::AbstractRNG)
    nf = size(features, 2)
    N = length(labels)

    best = NO_BEST
    best_val = -Inf

    if nsubfeatures > 0
        r = randperm(rng, nf)
        inds = r[1:nsubfeatures]
    else
        inds = 1:nf
    end

    for i in inds
        ord = sortperm(features[:,i])
        features_i = features[ord,i]
        labels_i = labels[ord]

        hist1 = _hist(labels_i, 1:0)
        hist2 = _hist(labels_i)
        N1 = 0
        N2 = N

        for (d, range) in UniqueRanges(features_i)
            value = _info_gain(N1, hist1, N2, hist2)
            if value > best_val
                best_val = value
                best = (i, d)
            end

            deltaN = length(range)

            _hist_shift!(hist2, hist1, labels_i, range)
            N1 += deltaN
            N2 -= deltaN
        end
    end
    return best
end

function _split_neg_z1_loss(labels::Vector, features::Matrix, weights::Vector)
    best = NO_BEST
    best_val = -Inf
    for i in 1:size(features,2)
        domain_i = sort(unique(features[:,i]))
        for thresh in domain_i[2:end]
            cur_split = features[:,i] .< thresh
            value = _neg_z1_loss(labels[cur_split], weights[cur_split]) + _neg_z1_loss(labels[!cur_split], weights[!cur_split])
            if value > best_val
                best_val = value
                best = (i, thresh)
            end
        end
    end
    return best
end

function build_stump(labels::Vector, features::Matrix, weights=[0];
                     rng=Base.GLOBAL_RNG)
    S = _split(labels, features, 0, weights, rng)
    if S == NO_BEST
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:, id] .< thresh
    return Node(id, thresh,
                Leaf(majority_vote(labels[split]), labels[split]),
                Leaf(majority_vote(labels[!split]), labels[!split]))
end


apply_tree(leaf::Leaf, feature::Vector) = leaf.majority

function apply_tree(tree::Node, features::Vector)
    if tree.featval == nothing
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree(tree::LeafOrNode, features::Matrix)
    N = size(features,1)
    predictions = Array(Any,N)
    for i in 1:N
        predictions[i] = apply_tree(tree, squeeze(features[i,:],1))
    end
    if typeof(predictions[1]) <: Float64
        return float(predictions)
    else
        return predictions
    end
end

"""
    apply_tree_proba(::Node, features, col_labels::Vector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix.
"""
apply_tree_proba(leaf::Leaf, features::Vector, labels) =
    compute_probabilities(labels, leaf.values)

function apply_tree_proba(tree::Node, features::Vector, labels)
    if tree.featval === nothing
        return apply_tree_proba(tree.left, features, labels)
    elseif features[tree.featid] < tree.featval
        return apply_tree_proba(tree.left, features, labels)
    else
        return apply_tree_proba(tree.right, features, labels)
    end
end

apply_tree_proba(tree::Node, features::Matrix, labels) =
    stack_function_results(row -> apply_tree_proba(tree, row, labels), features)


function build_adaboost_stumps(labels::Vector, features::Matrix, niterations::Integer; rng=Base.GLOBAL_RNG, subsamp = 0.7)
    N = length(labels)
    weights = ones(N)/N
    stumps = Node[]
    coeffs = Float64[]
    n_sub = round(Int, N*subsamp)

    for i in 1:niterations
        indcs = sample(1:N, n_sub, replace = false)

        new_stump = build_stump(labels[indcs], features[indcs, :], weights[indcs]; rng=rng)
        predictions = apply_tree(new_stump, features[indcs, :])
        err = _weighted_error(labels[indcs], predictions, weights[indcs])
        new_coeff = 0.5 * log((1.0 + err) / (1.0 - err))

        println("The error is $err and the new_coeff is $new_coeff")
        correct = labels[indcs] .== predictions
        matches = indcs[correct]
        non_matches = setdiff(indcs, matches)
        println(matches)
        println(non_matches)
        weights[non_matches] *= exp(new_coeff)
        weights[matches] *= exp(-new_coeff)
        weights[indcs] /= sum(weights[indcs])


        push!(coeffs, new_coeff)
        push!(stumps, new_stump)
        if err < 1e-6
            break
        end
    end
    return (Ensemble(stumps), coeffs)
end

function apply_adaboost_stumps(stumps::Ensemble, coeffs::Vector{Float64}, features::Vector)
    nstumps = length(stumps)
    counts = Dict()
    for i in 1:nstumps
        prediction = apply_tree(stumps.trees[i], features)
        counts[prediction] = get(counts, prediction, 0.0) + coeffs[i]
    end
    top_prediction = stumps.trees[1].left.majority
    top_count = -Inf
    for (k,v) in counts
        if v > top_count
            top_prediction = k
            top_count = v
        end
    end
    return top_prediction
end

function apply_adaboost_stumps(stumps::Ensemble, coeffs::Vector{Float64}, features::Matrix)
    N = size(features,1)
    predictions = Array(Any,N)
    for i in 1:N
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, squeeze(features[i,:],1))
    end
    return predictions
end

"""    apply_adaboost_stumps_proba(stumps::Ensemble, coeffs, features, labels::Vector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
function apply_adaboost_stumps_proba(stumps::Ensemble, coeffs::Vector{Float64},
                                     features::Vector, labels::Vector)
    votes = [apply_tree(stump, features) for stump in stumps.trees]
    compute_probabilities(labels, votes, coeffs)
end

function apply_adaboost_stumps_proba(stumps::Ensemble, coeffs::Vector{Float64},
                                    features::Matrix, labels::Vector)
    stack_function_results(row->apply_adaboost_stumps_proba(stumps, coeffs, row, labels), features)
end
