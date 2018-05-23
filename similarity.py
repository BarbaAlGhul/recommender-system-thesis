import numpy as np
import load_movies
from keras.models import load_model

EPSILON = 1e-07


def cosine(x, y):
    dot_pdt = np.dot(x, y.T)
    norms = np.linalg.norm(x) * np.linalg.norm(y)
    return dot_pdt / (norms + EPSILON)


# Computes cosine similarities between x and all item embeddings
# Calcula as 'cosine similarities' entre x e todos os embeddings dos items
def cosine_similarities(x, embeddings):
    dot_pdts = np.dot(embeddings, x)
    norms = np.linalg.norm(x) * np.linalg.norm(embeddings, axis=1)
    return dot_pdts / (norms + EPSILON)


# Computes euclidean distances between x and all item embeddings
# calcula as distancias euclidianas entre x e todos os embeddings dos items
def euclidean_distances(x, embeddings):
    return np.linalg.norm(embeddings - x, axis=1)


# Computes top_n most similar items to an index
# calcula os primeiros n itens similares ao índice
def most_similar(idx, embeddings, items, top_n=20, mode=None):
    namesdic = {row[1]['movieId']: row[1]['title']
                                   + '(' + str(int(row[1]['year'])) + ') ' for row in movies.iterrows()}
    if mode == 'euclidean':
        dists = euclidean_distances(embeddings[idx], embeddings)
        order = dists.argsort()
        order = [x for x in order if x != idx]
        order = order[:top_n]
        return list(zip([namesdic[x] for x in order], dists[order]))
    else:
        dists = cosine_similarities(embeddings[idx], embeddings)
        order = (-dists).argsort()
        order = [x for x in order if x != idx]
        order = order[:top_n]
        return list(zip([namesdic[x] for x in order], dists[order]))


movies = load_movies.load()

# carrega o modelo
model = load_model('models/NOME_DO_MODELO')
weights = model.get_weights()
item_embeddings = weights[1]

# entra com o código de um filme
movie_code = 'ID DO FILME'
print("Itens próximos a " + str(movies.title[movie_code]) + '(' + str(int(movies.year[movie_code])) + ')' + ":")
for title, dist in most_similar(2744, item_embeddings, movies, mode='euclidean'):
    print(title, dist)
