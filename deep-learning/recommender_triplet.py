import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import time
from time import localtime, strftime
import send_email
import pickle
import load_movies
import load_ratings
import losses as ls
import sample_triplets as st
import evaluate as evl


# captura o tempo agora, somente para informação e análise dos resultados
date_now = strftime("%d/%m/%Y %H:%M:%S", localtime())

# carrega o dataset de ratings
ratings = load_ratings.load('../')
# carrega o dataset de filmes
movies = load_movies.load('../')

# divide o dataset em 80% para treinamento e 20% para teste
train, test = train_test_split(ratings, test_size=0.3, random_state=0)

# separa somente as ratings >= 4 nos sets de train e test
pos_train = train.query('rating >= 4')
pos_test = test.query('rating >= 4')

n_users, n_movies = len(ratings.userId.unique()), len(ratings.movieId.unique())

movie_dim = 64
user_dim = 64
hidden_size = 128

# cria as camadas da rede neural
movie_input = layers.Input(shape=(1,), name='Movie')
user_input = layers.Input(shape=(1,), name='User')
positive_movie_input = layers.Input(shape=(1,), name='Positive-Movie')
negative_movie_input = layers.Input(shape=(1,), name='Negative-Movie')

movie_embedding = layers.Embedding(input_dim=n_movies,
                                   input_length=1,
                                   output_dim=movie_dim,
                                   name='Movie-Embedding')
user_embedding = layers.Embedding(input_dim=n_users,
                                  input_length=1,
                                  output_dim=user_dim,
                                  name='User-Embedding')

movie_vec_positive = layers.Flatten()(movie_embedding(positive_movie_input))
movie_vec_negative = layers.Flatten()(movie_embedding(negative_movie_input))
user_vec = layers.Flatten()(user_embedding(user_input))

positive_embeddings = layers.Concatenate()([user_vec, movie_vec_positive])
positive_embeddings = layers.Dropout(0.2)(positive_embeddings)
negative_embeddings = layers.Concatenate()([user_vec, movie_vec_negative])
negative_embeddings = layers.Dropout(0.2)(negative_embeddings)

layers_seq = keras.Sequential()
layers_seq.add(layers.Dense(hidden_size, input_dim=movie_dim+user_dim, activation='relu'))
layers_seq.add(layers.Dropout(0.2))
for i in range(0, 2):
    layers_seq.add(layers.Dense(hidden_size, activation='relu'))
    layers_seq.add(layers.Dropout(0.1))
layers_seq.add(layers.Dense(1, activation='relu'))

positive_similarity = layers_seq(positive_embeddings)
negative_similarity = layers_seq(negative_embeddings)

triplet_loss = layers.Lambda(ls.margin_comparator_loss,
                             output_shape=(1,),
                             name='Comparator-Loss')([positive_similarity, negative_similarity])

model = keras.Model(inputs=[user_input, positive_movie_input, negative_movie_input], outputs=triplet_loss)

# The match-score model, only use at inference to rank items for a given
# model: the model weights are shared with the triplet_model therefore
# we do not need to train it and therefore we do not need to plug a loss
# and an optimizer.

match_model = keras.Model(inputs=[user_input, positive_movie_input], outputs=positive_similarity)

model.compile(optimizer='adam', loss=ls.identity_loss)

# cria uma imagem do modelo da rede
plot_model(model, to_file='model_deep_learning_triplet.png', show_shapes=True)

# imprime o resumo dos modelos em um arquivo
with open('model_summary_deep_learning_triplet.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
f.close()

# variável para guardar o número de epochs
epochs = 16

# maior ID entre os filmes, para o treinamento
max_movie_id = max(train.movieId.max(), test.movieId.max())

# salva os modelos de acordo com o callback do Keras
save_path = '../models'
my_time = time.strftime("%Y_%m_%d_%H_%M")
model_name = 'deep_learning_triplet_' + my_time
full_name = save_path + '/' + model_name + '.h5'
m_check = ModelCheckpoint(full_name, monitor='val_loss', save_best_only=True)

fake_y = np.ones(len(pos_train.userId))

# sanity check
test_map = evl.mean_average_precision(match_model, train, test)
test_ndcg = evl.normalized_dcg(match_model, train, test)
test_auc = evl.roc_auc(match_model, train, test)
print("Check MAP: %0.4f" % test_map)
print("Check NDCG: %0.4f" % test_ndcg)
print("Check ROC_AUC: %0.4f" % test_auc)

# começa a contar o tempo do treinamento
start_time = time.time()

for i in range(epochs):

    triplet_inputs = st.sample_triplets(pos_train, max_movie_id, random_seed=i)

    # faz o treinamento do modelo
    history = model.fit(triplet_inputs,
                        fake_y,
                        epochs=1,
                        batch_size=64,
                        verbose=2,
                        shuffle=True,
                        validation_split=0.1,
                        callbacks=[m_check])

    print('Epoch %d/%d: ' % (i + 1, epochs))

# mostra o tempo to treinamento no formato hh:mm:ss
seconds = (time.time() - start_time)
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print('%02d:%02d:%02d' % (h, m, s))

test_map = evl.mean_average_precision(match_model, train, test)
test_ndcg = evl.normalized_dcg(match_model, train, test)
test_auc = evl.roc_auc(match_model, train, test)
print("MAP: %0.4f" % test_map)
print("NDCG: %0.4f" % test_ndcg)
print("ROC_AUC: %0.4f" % test_auc)
'''
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
# fig1.savefig('training_loss_deep_learning.png', dpi=200)
'''

'''
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
'''
# manda um email com os resultados da execução, passando como parâmetro arquivos para mandar como anexo
# send_email.send(['training_loss_deep_learning.png', 'model_deep_learning.png', 'model_summary_deep_learning.txt'])
