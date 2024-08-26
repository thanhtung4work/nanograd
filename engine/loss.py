def cross_entropy(y_pred, y_truth, n_classes):
    def _get_list():
        ce_list = [-1] * n_classes
        ce_list[y_truth] = 1
        return ce_list
    y_truth = _get_list()
    exps = [y.exp() for y in y_pred]
    sum_exps = sum(exps)
    softmax_values = [(exp / sum_exps) for exp in exps]
    
    loss = 0
    for yp, yt in zip(softmax_values, y_truth):
        loss += -1 * yt * yp.log()
    return loss