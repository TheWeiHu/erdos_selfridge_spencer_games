optimal_k = n_tokens // 2
    # if n_tokens % 2 == 0 and n_buckets >= optimal_k:
    #     return 3 ** optimal_k
    # if n_tokens % 2 == 1 and n_buckets >= optimal_k + 1:
    #     return 3 ** (optimal_k) * 2