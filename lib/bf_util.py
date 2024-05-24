import math


def get_fpr(n_items, bf_size):
    m = bf_size * 8
    # 粗略估算最佳 k 值
    best_k = round((m / n_items) * math.log(2))
    optimal_error_rate = (1 - math.exp(-best_k * n_items / m)) ** best_k
    return optimal_error_rate
