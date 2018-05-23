import pandas as pd


def load(path=''):
    ratings = pd.read_csv(path+'ml-latest-small/ratings.csv')

    # remove todos os NaNs
    ratings.dropna(inplace=True)

    # organiza as informações dos dois arquivos
    ratings.sort_values(by='movieId', inplace=True)
    ratings.reset_index(inplace=True, drop=True)
    return ratings
