import metrics as mt
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import numpy as np


def mean_average_precision(match_model, train_data, test_data):
    max_user_id = max(train_data.userId.max(), test_data.userId.max())
    max_movie_id = max(train_data.movieId.max(), test_data.movieId.max())
    user_map_score = []
    for user_id in range(0, max_user_id + 1):
        pos_movie_train = train_data[train_data.userId == user_id]
        pos_movie_test = test_data[test_data.userId == user_id]
        # considera todos os itens já classificados
        movies_ids = np.arange(0, max_movie_id + 1)
        movies_to_rank = np.setdiff1d(movies_ids, pos_movie_train.movieId.values)
        # Ground truth: retorna 1 para cada item presente no set de teste, retorna 0 caso contrário
        expected = np.in1d(movies_to_rank, pos_movie_test.movieId.values)

        if np.sum(expected) >= 1:
            # pelo menos um resultado positivo para avaliar
            repeated_user_id = np.empty_like(movies_to_rank)
            repeated_user_id.fill(user_id)
            predicted = match_model.predict([repeated_user_id, movies_to_rank],
                                            batch_size=4096)
            user_map_score.append(average_precision_score(expected, predicted))

    return np.mean(user_map_score, dtype=np.float32)


def normalized_dcg(match_model, train_data, test_data):
    max_user_id = max(train_data.userId.max(), test_data.userId.max())
    max_movie_id = max(train_data.movieId.max(), test_data.movieId.max())
    user_ndcg_score = []
    for user_id in range(0, max_user_id + 1):
        pos_movie_train = train_data[train_data.userId == user_id]
        pos_movie_test = test_data[test_data.userId == user_id]
        # considera todos os itens já classificados
        movies_ids = np.arange(0, max_movie_id + 1)
        movies_to_rank = np.setdiff1d(movies_ids, pos_movie_train.movieId.values)
        # Ground truth: retorna 1 para cada item presente no set de teste, retorna 0 caso contrário
        expected = np.in1d(movies_to_rank, pos_movie_test.movieId.values)

        if np.sum(expected) >= 1:
            # pelo menos um resultado positivo para avaliar
            repeated_user_id = np.empty_like(movies_to_rank)
            repeated_user_id.fill(user_id)
            predicted = match_model.predict([repeated_user_id, movies_to_rank],
                                            batch_size=4096)
            user_ndcg_score.append(mt.ndcg_at_k(predicted, 20))

    return np.mean(user_ndcg_score, dtype=np.float32)


def roc_auc(match_model, train_data, test_data):
    max_user_id = max(train_data.userId.max(), test_data.userId.max())
    max_movie_id = max(train_data.movieId.max(), test_data.movieId.max())
    user_score = []
    for user_id in range(0, max_user_id + 1):
        pos_movie_train = train_data[train_data.userId == user_id]
        pos_movie_test = test_data[test_data.userId == user_id]
        # considera todos os itens já classificados
        movies_ids = np.arange(0, max_movie_id + 1)
        movies_to_rank = np.setdiff1d(movies_ids, pos_movie_train.movieId.values)
        # Ground truth: retorna 1 para cada item presente no set de teste, retorna 0 caso contrário
        expected = np.in1d(movies_to_rank, pos_movie_test.movieId.values)

        if np.sum(expected) >= 1:
            # pelo menos um resultado positivo para avaliar
            repeated_user_id = np.empty_like(movies_to_rank)
            repeated_user_id.fill(user_id)
            predicted = match_model.predict([repeated_user_id, movies_to_rank],
                                            batch_size=4096)
            user_score.append(roc_auc_score(expected, predicted))

    return np.mean(user_score, dtype=np.float32)

