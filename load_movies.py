import pandas as pd


def load(path=''):
    # carrega os filmes
    movies = pd.read_csv(path+'ml-latest-small/movies.csv')
    # separa o ano do filme do título e cria uma coluna para esta informação
    movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
    movies.year = pd.to_datetime(movies.year, format='%Y')
    # como alguns resultados serão NaN, o tipo resultante será float
    movies.year = movies.year.dt.year
    movies.title = movies.title.str[:-7]

    # formata os gêneros dos filmes de forma adequada
    genres_unique = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()
    genres_unique = pd.DataFrame(genres_unique, columns=['genre'])
    movies = movies.join(movies.genres.str.get_dummies().astype(bool))
    movies.drop('genres', inplace=True, axis=1)

    # remove todos os NaNs
    movies.dropna(inplace=True)

    # mapeia os ids dos filmes e faz eles começarem do 0 até o número total
    inverse_movies = {val: i for i, val in enumerate(movies.movieId.unique())}
    movies.movieId = movies.movieId.map(inverse_movies)

    # organiza as informações do arquivo
    movies.sort_values(by='movieId', inplace=True)
    movies.reset_index(inplace=True, drop=True)
    return movies
