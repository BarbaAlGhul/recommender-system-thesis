import numpy as np


def sample_triplets(pos_data, max_item_id, random_seed=0):
    """Sample negatives at random"""
    rng = np.random.RandomState(random_seed)
    user_ids = pos_data.userId.values
    pos_movies_ids = pos_data.movieId.values

    neg_item_ids = rng.randint(low=1, high=max_item_id + 1,
                               size=len(user_ids))

    return [pos_movies_ids, neg_item_ids, user_ids]
