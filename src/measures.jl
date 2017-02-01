# measures.jl
function _hist_add!{T}(counts::Dict{T,Int}, labels::Vector{T}, region::Range1{Int})
    for i in region
        lbl = labels[i]
        counts[lbl] = get(counts, lbl, 0) + 1
    end
    return counts
end


function _hist_sub!{T}(counts::Dict{T,Int}, labels::Vector{T}, region::Range1{Int})
    for i in region
        lbl = labels[i]
        counts[lbl] -= 1
    end
    return counts
end


function _hist_shift!{T}(counts_from::Dict{T,Int}, counts_to::Dict{T,Int}, labels::Vector{T}, region::Range1{Int})
    for i in region
        lbl = labels[i]
        counts_from[lbl] -= 1
        counts_to[lbl] = get(counts_to, lbl, 0) + 1
    end
end


_hist{T}(labels::Vector{T}, region::Range1{Int} = 1:endof(labels)) = _hist_add!(Dict{T,Int}(), labels, region)


function _info_gain{T}(N1::Int, counts1::Dict{T,Int}, N2::Int, counts2::Dict{T,Int})
    N = N1 + N2
    H = - N1/N * _set_entropy(counts1, N1) - N2/N * _set_entropy(counts2, N2)
    return H
end


function _neg_z1_loss{T<:Real}(labels::Vector, weights::Vector{T})
    missmatches = labels .!= majority_vote(labels)
    loss = sum(weights[missmatches])
    return -loss
end


function _weighted_error{T<:Real}(actual::Vector, predicted::Vector, weights::Vector{T})
    mismatches = actual .!= predicted
    err = sum(weights[mismatches]) / sum(weights)
    return err
end




function majority_vote(labels::Vector)
    if length(labels) == 0
        return nothing
    end
    counts = _hist(labels)
    top_vote = labels[1]
    top_count = -1
    for (k,v) in counts
        if v > top_count
            top_vote = k
            top_count = v
        end
    end
    return top_vote
end



function adaboost_train_error(y, X, model, coeffs)
    y_hat = apply_adaboost_stumps(model, coeffs, X)
    return mean(y .!= y_hat)
end



function matthews_corrcoef(y_hat, y)
    n = length(y)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i = 1:n
        if y_hat[i] == 1
            if y[i] == 1
                tp += 1
            else
                fp += 1
            end
        elseif y_hat[i] == 0
            if y[i] == 0
                tn += 1
            else
                fn += 1
            end
        end
    end
    mcc = (tp * tn - fp * fn)/sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc
end
