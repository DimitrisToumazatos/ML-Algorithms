import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from keras import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

list1 = [2000, 4000, 6000, 8000, 10000] # Number of examples from 1 category

for k in list1:
    
    ###############################################################################################
    (x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data()
    temp1 =  x_train_imdb[divmod(len(x_train_imdb), 2)[0]: divmod(len(x_train_imdb), 2)[0] + k]
    x_train_imdb = x_train_imdb[:k]
    x_train_imdb = np.append(temp1, x_train_imdb)
    temp1 = y_train_imdb[divmod(len(y_train_imdb), 2)[0]: divmod(len(y_train_imdb), 2)[0] + k]
    y_train_imdb = y_train_imdb[:k]
    y_train_imdb = np.append(temp1, y_train_imdb)
    

    word_index = tf.keras.datasets.imdb.get_word_index()
    index2word = dict((i + 3, word) for (word, i) in word_index.items())
    index2word[0] = '[pad]'
    index2word[1] = '[bos]'
    index2word[2] = '[oov]'
    x_train_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train_imdb])
    x_test_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test_imdb])



    train_doc_length = 0
    for doc in tqdm(x_train_imdb):
        tokens = str(doc).split()
        train_doc_length += len(tokens)

    print('\nTraining data average document length =', (train_doc_length 
                                                        / len(x_train_imdb)))

    VOCAB_SIZE = 100000
    SEQ_MAX_LENGTH = 250
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, 
                                                    output_mode='int', 
                                                    ngrams=1, name='vector_text',
                                                    output_sequence_length=SEQ_MAX_LENGTH)
    with tf.device('/CPU:0'):
        vectorizer.adapt(x_train_imdb)


    vector_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            vectorizer
    ])
    vector_model.predict([['awesome movie']])

    dummy_embeddings = tf.keras.layers.Embedding(1000, 5)
    dummy_embeddings(tf.constant([1, 2, 3])).numpy()

    def get_rnn(num_layers=1, emb_size=64, h_size=64):
        inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='txt_input') # ['awesome movie']
        x = vectorizer(inputs) # [1189, 18, 0, 0, 0, 0, ...]
        x = tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                    output_dim=emb_size, name='word_embeddings',
                                    mask_zero=True)(x)
        for n in range(num_layers):
            if n != num_layers - 1:
                x = tf.keras.layers.SimpleRNN(units=h_size, 
                                            name=f'rnn_cell_{n}', 
                                            return_sequences=True)(x)
            else:
                x = tf.keras.layers.SimpleRNN(units=h_size, name=f'rnn_cell_{n}')(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        o = tf.keras.layers.Dense(units=1, activation='sigmoid', name='lr')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=o, name='simple_rnn')

    imdb_rnn = get_rnn()

    imdb_rnn.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['binary_accuracy'])
    epochs = [2, 4, 6, 8, 10]
    for e in epochs:
        imdb_rnn.fit(x=x_train_imdb, y=y_train_imdb,
                    epochs=e, verbose=1, batch_size=32)
        print("Statistics for " + str(2 * k) + " train examples and " + str(e) + " epochs are the following:")
        loss1, accuracy = imdb_rnn.evaluate(x_test_imdb[:k], np.array(y_test_imdb[:k]))
        truePositive = round(accuracy * k)
        loss2, accuracy = imdb_rnn.evaluate(x_test_imdb[k:], np.array(y_test_imdb[k:]))
        

        # print train statistics
        print("Loss: " + str((loss1 + loss2) / 2))