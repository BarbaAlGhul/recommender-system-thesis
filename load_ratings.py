import pandas as pd


def load(path=''):
    ratings = pd.read_csv(path+'ml-latest-small/ratings.csv')

    # remove todos os NaNs
    ratings.dropna(inplace=True)

    # mapeia os ids dos filmes e faz eles começarem do 0 até o número total
    inverse_movies_ratings = {val: i for i, val in enumerate(ratings.movieId.unique())}
    ratings.movieId = ratings.movieId.map(inverse_movies_ratings)

    # organiza as informações do arquivo
    ratings.sort_values(by='movieId', inplace=True)
    ratings.reset_index(inplace=True, drop=True)
    return ratings
