import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import plot_model
from keras.constraints import non_neg

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import time


# carrega o dataset
dataset = pd.read_csv("ml-latest-small/ratings.csv")

# atribui um único número (entre 0 e o número de usuários) para cada usuário e faz o mesmo para os filmes
dataset.userId = dataset.userId.astype('category').cat.codes.values
dataset.movieId = dataset.movieId.astype('category').cat.codes.values

# divide o dataset em 80% para treinamento e 20% para teste
train, test = train_test_split(dataset, test_size=0.2)

# o método len() retorna o número de elementos em uma lista
# então o número de usuários e filmes únicos estão sendo colocados em uma lista
n_users, n_movies = len(dataset.userId.unique()), len(dataset.movieId.unique())
n_latent_factors = 3

# cria as camadas, usando uma matrix de fatorização não negativa
movie_input = keras.layers.Input(shape=[1], name='Movie')
movie_embedding = keras.layers.Embedding(
    n_movies + 1, n_latent_factors, name='Movie-Embedding', embeddings_constraint=non_neg())(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

user_input = keras.layers.Input(shape=[1], name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(
    n_users + 1, n_latent_factors, name='User-Embedding', embeddings_constraint=non_neg())(user_input))

prod = keras.layers.dot([movie_vec, user_vec], axes=1, name='DotProduct')
model = keras.Model([user_input, movie_input], prod)
model.compile('adam', 'mean_squared_error')

# cria uma imagem do modelo da rede
plot_model(model, to_file='model.png', show_shapes=True)

# imprime o resumo do modelo em um arquivo
with open('model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
f.close()

# começa a contar o tempo do treinamento
start_time = time.time()

# faz o treinamento do modelo
history = model.fit([train.userId, train.movieId], train.rating, epochs=100, verbose=1)


# plota um gráfico da perda do treinamento e salva em uma imagem
pd.Series(history.history['loss']).plot(logy=True)
plt.xlabel('Epoch')
plt.ylabel('Training Error')
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('training_loss.png', dpi=200)

# valores corretos baseados no modelo
y_hat = np.round(model.predict([test.userId, test.movieId]), 0)
# valores estimados
y_true = test.rating
# imprime o "mean absolute error"
print(mean_absolute_error(y_true, y_hat))

# mostra o tempo to treinamento no formato hh:mm:ss
seconds = (time.time() - start_time)
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print('%02d:%02d:%02d' % (h, m, s))

movie_embedding_learnt = model.get_layer(name='Movie-Embedding').get_weights()[0]
print(pd.DataFrame(movie_embedding_learnt).describe())

user_embedding_learnt = model.get_layer(name='User-Embedding').get_weights()[0]
print(pd.DataFrame(user_embedding_learnt).describe())

# imprime os resultados em um arquivo
with open('results.txt', 'w') as fr:
    fr.write('Tempo de execução: ' + str('%02d:%02d:%02d' % (h, m, s)) + '\n')
    fr.write('\n' + 'Mean Absolute Error: ' + str(mean_absolute_error(y_true, y_hat)) + '\n')
    fr.write('\n' + 'Resultado do aprendizado dos filmes: ' + '\n' +
             str(pd.DataFrame(movie_embedding_learnt).describe()) + '\n')
    fr.write('\n' + 'Resultado do aprendizado dos usuários: ' + '\n' +
             str(pd.DataFrame(user_embedding_learnt).describe()) + '\n')
fr.close()
