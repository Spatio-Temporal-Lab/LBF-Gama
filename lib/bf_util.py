import math
from pybloom_live import BloomFilter


def get_fpr(n_items, bf_size):
    m = bf_size * 8
    n_items = max(1, n_items)
    return max(1e-8, 0.5 ** (m * math.log(2) / n_items))


def create_bloom_filter(dataset, bf_size):
    n_items = len(dataset)
    # print('n_items = ', n_items)
    # print('bf_size = ', bf_size)

    # # 创建布隆过滤器
    bloom_filter = BloomFilter(capacity=max(1, n_items), error_rate=get_fpr(n_items, bf_size))
    for data in dataset:
        bloom_filter.add(data)

    return bloom_filter


def create_bloom_filter_in_bits(dataset, bf_size):
    n_items = len(dataset)

    # # 创建布隆过滤器
    bloom_filter = BloomFilter(capacity=max(1, n_items), error_rate=max(1e-8, 0.5 ** (bf_size * math.log(2) / n_items)))
    for data in dataset:
        bloom_filter.add(data)

    return bloom_filter
