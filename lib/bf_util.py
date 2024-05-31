import math


def get_fpr(n_items, bf_size):
    m = bf_size * 8
    n_items = max(1, n_items)
    return max(1e-8, 0.5 ** (m * math.log(2) / n_items))
