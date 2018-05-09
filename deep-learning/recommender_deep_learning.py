import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from time import localtime, strftime
import send_email
import pickle


# captura o tempo agora, somente para informação e análise dos resultados
date_now = strftime("%d/%m/%Y %H:%M:%S", localtime())

# carrega o dataset
dataset = pd.read_csv('../ml-latest-small/ratings.csv')

# atribui um único número (entre 0 e o número de usuários) para cada usuário e faz o mesmo para os filmes
dataset.userId = dataset.userId.astype('category').cat.codes.values
dataset.movieId = dataset.movieId.astype('category').cat.codes.values

# divide o dataset em 80% para treinamento e 20% para teste
train, test = train_test_split(dataset, test_size=0.2, random_state=0)

n_users, n_movies = len(dataset.userId.unique()), len(dataset.movieId.unique())

embedding_size = 10

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

movie_vec = layers.Reshape([embedding_size])(movie_embedding)
user_vec = layers.Reshape([embedding_size])(user_embedding)

input_vecs = layers.Concatenate()([user_vec, movie_vec])

dense_1 = layers.Dense(128, activation='relu')(input_vecs)
dropout_1 = layers.Dropout(0.3)(dense_1)
dense_2 = layers.Dense(1)(dense_1)

model = keras.Model(inputs=[user_input, movie_input], outputs=dense_2)
model.compile(optimizer='adam', loss='mean_squared_error')

# cria uma imagem do modelo da rede
plot_model(model, to_file='model_deep_learning.png', show_shapes=True)

# imprime o resumo do modelo em um arquivo
with open('model_summary_deep_learning.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
f.close()

# variável para guardar o número de epochs
epochs = 25

# salva os modelos de acordo com o callback do Keras
save_path = '../models'
my_time = time.strftime("%Y_%m_%d_%H_%M")
model_name = 'deep_learning_' + my_time
full_name = save_path + '/' + model_name + '.h5'
m_check = ModelCheckpoint(full_name, monitor='val_loss', save_best_only=True)

# começa a contar o tempo do treinamento
start_time = time.time()

# faz o treinamento do modelo
history = model.fit([train.userId, train.movieId],
                    train.rating,
                    epochs=epochs,
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
pd.Series(val_loss).plot(label='Training val_loss')
plt.title('Perda do treinamento')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('training_loss_deep_learning.png', dpi=200)

# valores baseados no modelo
y_pred = np.round(model.predict([test.userId, test.movieId]), 0)
# valores estimados
y_true = test.rating
# imprime o erro
print('Erro: ' + str(mean_squared_error(y_true, y_pred)))
# imprime a previsão do erro
print('Previsão do erro: ' + str(mean_squared_error(y_true, model.predict([test.userId, test.movieId]))))

# imprime os resultados em um arquivo
with open('results.txt', 'w') as fr:
    fr.write('Data de treinamento da rede: ' + date_now + '\n')
    fr.write('\n' + 'Tempo de execução: ' + str('%02d:%02d:%02d' % (h, m, s)) + '\n')
    fr.write('\n' + 'Mean Squared Error: ' + str(mean_squared_error(y_true, y_pred)) + '\n')
    fr.write('\n' + 'Mean Squared Error Prediction: ' + str(
        mean_squared_error(y_true, model.predict([test.userId, test.movieId]))) + '\n')
    fr.write('\n' + 'Número de Epochs da rede: ' + str(epochs) + '\n')
fr.close()

# manda um email com os resultados da execução, passando como parâmetro arquivos para mandar como anexo
send_email.send(['training_loss_deep_learning.png', 'model_deep_learning.png', 'model_summary_deep_learning.txt'])
