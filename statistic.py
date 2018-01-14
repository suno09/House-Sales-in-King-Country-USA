def adjusted_r2(r2: float, nbr_features: int, sample_size: int):
    return 1. - (1. - r2) * (sample_size - 1) / (sample_size - nbr_features - 1)
