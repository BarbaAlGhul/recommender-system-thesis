import numpy as np
from sklearn.preprocessing import scale
import load_ratings
import load_movies
from keras.models import load_model


def recommend(movies, ratings, model, user_id, top_n=10):
    # os filmes que os usuários não classificaram serão considerados não assistidos para fins de recomendação
    # criando uma lista dos filmes que o usuário classificou
    rated_movies = list(ratings[ratings['userId'] == user_id]['movieId'])
    # cria uma lista com os filmes não classificados
    movies_ids = list(filter(lambda x: x not in rated_movies, movies.movieId))
    print("Usuário " + str(user_id) + " assistiu " + str(len(rated_movies)) + " filmes. " +
          "Calculando ratings para outros " + str(len(movies_ids)) + " filmes.")

    movies['scaled_year'] = scale(movies['year'].astype('float64'))

    movies_ids = np.array(movies_ids)
    user = np.zeros_like(movies_ids)
    user[:] = user_id

    rating_preds = model.predict([user, movies_ids])
    movies_ids = np.argsort(rating_preds[:, 0])[::-1].tolist()
    rec_items = movies_ids[:top_n]
    return [(movies['title'][movie], str(int(movies['year'][movie])), rating_preds[movie][0]) for movie in rec_items]


movies = load_movies.load()
ratings = load_ratings.load()

# carrega o modelo
model = load_model('models/autoencoder_2018_05_23_10_52.h5')

recommendations = recommend(movies, ratings, model, 3)
for elem in recommendations:
    print(elem)
