def auroc(scores, labels):
    if len(set(labels)) < 2:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = 0
    r = 1
    for _, y in pairs:
        if y == 1:
            rank_sum += r
        r += 1
    u = rank_sum - pos * (pos + 1) / 2
    return u / (pos * neg)
