import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import time
from time import localtime, strftime
import send_email
import pickle
import load_movies
import load_ratings


# captura o tempo agora, somente para informação e análise dos resultados
date_now = strftime("%d/%m/%Y %H:%M:%S", localtime())

# carrega o dataset de ratings
ratings = load_ratings.load('../')
# carrega o dataset de filmes
movies = load_movies.load('../')

# divide o dataset em 80% para treinamento e 20% para teste
train, test = train_test_split(ratings, test_size=0.2, random_state=0)

n_users, n_movies = len(ratings.userId.unique()), len(ratings.movieId.unique())

embedding_size = 16

# cria as camadas da rede neural
movie_input = layers.Input(shape=[1], name='Movie')
user_input = layers.Input(shape=[1], name='User')

movie_embedding = layers.Embedding(input_dim=n_movies,
                                   input_length=1,
                                   output_dim=embedding_size,
                                   name='Movie-Embedding')(movie_input)
user_embedding = layers.Embedding(input_dim=n_users,
                                  input_length=1,
                                  output_dim=embedding_size,
                                  name='User-Embedding')(user_input)

# movie_vec = layers.Reshape([embedding_size])(movie_embedding)
# user_vec = layers.Reshape([embedding_size])(user_embedding)

movie_vec = layers.Flatten()(movie_embedding)
user_vec = layers.Flatten()(user_embedding)

input_vecs = layers.Concatenate()([user_vec, movie_vec])
encoded = layers.Dense(64, activation='relu')(input_vecs)
encoded = layers.Dropout(0.5)(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dropout(0.5)(decoded)
decoded = layers.Dense(32, activation='relu')(decoded)
decoded = layers.Dense(1, activation='relu')(decoded)

model = keras.Model(inputs=[user_input, movie_input], outputs=decoded)
model.compile(optimizer='adam', loss='mae')

# cria uma imagem do modelo da rede
plot_model(model, to_file='model_autoencoder.png', show_shapes=True)

# imprime o resumo do modelo em um arquivo
with open('model_summary_autoencoder.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
f.close()

# variável para guardar o número de epochs
epochs = 20

# salva os modelos de acordo com o callback do Keras
save_path = '../models'
my_time = time.strftime("%Y_%m_%d_%H_%M")
model_name = 'autoencoder_' + my_time
full_name = save_path + '/' + model_name + '.h5'
m_check = ModelCheckpoint(full_name, monitor='val_loss', save_best_only=True)

# começa a contar o tempo do treinamento
start_time = time.time()

# faz o treinamento do modelo
history = model.fit([train.userId, train.movieId],
                    train.rating,
                    epochs=epochs,
                    batch_size=64,
                    verbose=2,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[m_check])

# mostra o tempo to treinamento no formato hh:mm:ss
seconds = (time.time() - start_time)
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print('%02d:%02d:%02d' % (h, m, s))

# salva o treinamento
history_name = 'dense_' + my_time
with open('../histories/' + history_name + '.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# plota um gráfico da perda em relação às epochs e depois salva em uma imagem
loss = history.history['loss']
val_loss = history.history['val_loss']
pd.Series(loss).plot(label='Training loss')
pd.Series(val_loss).plot(label='Validation loss')
plt.title('Perda do treinamento')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('training_loss_autoencoder.png', dpi=200)

# imprime a MSE e a MAE do teste e do treinamento
test_preds = model.predict([test.userId, test.movieId])
final_test_mse = "Final test MSE: %0.3f" % mean_squared_error(test_preds, test.rating)
final_test_mae = "Final test MAE: %0.3f" % mean_absolute_error(test_preds, test.rating)
print(final_test_mse)
print(final_test_mae)
train_preds = model.predict([train.userId, train.movieId])
final_train_mse = "Final train MSE: %0.3f" % mean_squared_error(train_preds, train.rating)
final_train_mae = "Final train MAE: %0.3f" % mean_absolute_error(train_preds, train.rating)
print(final_train_mse)
print(final_train_mae)

# imprime os resultados em um arquivo
with open('results.txt', 'w') as fr:
    fr.write('Data de treinamento da rede: ' + date_now + '\n')
    fr.write('\n' + 'Tempo de execução: ' + str('%02d:%02d:%02d' % (h, m, s)) + '\n')
    fr.write('\n' + str(final_test_mse) + '\n')
    fr.write('\n' + str(final_test_mae) + '\n')
    fr.write('\n' + str(final_train_mse) + '\n')
    fr.write('\n' + str(final_train_mae) + '\n')
    fr.write('\n' + 'Número de Epochs da rede: ' + str(epochs) + '\n')
fr.close()

# manda um email com os resultados da execução, passando como parâmetro arquivos para mandar como anexo
send_email.send(['training_loss_autoencoder.png', 'model_autoencoder.png', 'model_summary_autoencoder.txt'])
