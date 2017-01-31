
function trapezoidal_area(x1, x2, y1, y2)
    base = abs(x1 - x2)
    height = (y1 + y2)/2
    soln = base * height
    soln
end


function auc(y, f, n_pos, n_neg)
    idx_order = sortperm(f, rev = true)
    y_sorted = y[idx_order]
    fp = 0
    tp = 0
    fp_prev = 0
    tp_prev = 0
    area = 0.0


    for i = 1:length(y_sorted)
        if i > 1 && f[i] ≠ f[i-1]
            area += trapezoidal_area(fp, fp_prev, tp, tp_prev)
            fp_prev = fp
            tp_prev = tp
        end
        if y_sorted[i] == 1
            tp += 1
        else
            fp += 1
        end
    end
    area += trapezoidal_area(n_neg, fp_prev, n_neg, tp_prev)
    area = area/(n_pos * n_neg)
    return area
end







function _binary_clf_curve(y_true, y_score)
    y_true = y_true .== 1       # make y_true a boolean vector
    desc_score_indices = sortperm(y_score, rev = true)

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = find(diff(y_score))
    threshold_idxs = push!(distinct_value_indices, length(y_score))

    tps = cumsum(y_true)[threshold_idxs]
    fps = threshold_idxs - tps
    return (fps, tps, y_score[threshold_idxs])
end



y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
y_score = [0.3, 0.2, 0.3, 0.23, 0.5, 0.34, 0.45, 0.54, 0.6, 0.7, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.6, 0.7, 0.5]

_binary_clf_curve(y, y_score)




function roc_curve(y_true, y_score)
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    fpr = fps/fps[end]
    tpr = tps/tps[end]
    return (fpr, tpr, thresholds)
end

roc_curve(y, y_score)










function sortperm2(x, y, rev = false)
    n = length(x)
    no_ties = n == length(Set(x))
    if no_ties
        res = sortperm(x, rev = rev)
    else
        ord1 = sortperm(x, rev = rev)
        x_sorted = x[ord1]
        for i = 1:(n-1)
            if x_sorted[i] == x_sorted[i+1]
                if rev && y[ord1][i] < y[ord1][i+1]
                    ord1[i], ord1[i+1] = ord1[i+1], ord1[i]
                elseif !rev && y[ord1][i] > y[ord1][i+1]
                    ord1[i], ord1[i+1] = ord1[i+1], ord1[i]
                end
            end
        end
        res = ord1
    end
    res
end

x1 = [4, 1, 1, 2, 3, 3]
y1 = [2, 3, 2, 5, 4, 3]

sortperm2(x1, y1)


function auc(x, y, reorder = false)
    direction = 1
    # if reorder
        # order = np.
end
