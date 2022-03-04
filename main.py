#!/usr/bin/env python
# coding: utf-8
import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

warnings.filterwarnings('ignore')
from skimage.transform import resize
from tensorflow.keras.models import Model

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Input, Softmax, RNN, Dense, Embedding, LSTM, Layer, Dropout, GRU

import flask
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template, jsonify

pretrained_img_model = DenseNet121(
    weights=r'pretrained_model/brucechou1983_CheXNet_Keras_0.3.0_weights.h5',
    classes=14, input_shape=(256, 256, 3))
model = Model(pretrained_img_model.input, pretrained_img_model.layers[-2].output)
model.save("CheXNet")

train_data = np.load(r"datasets/train.npy", allow_pickle=True)
test_data = np.load(r"datasets/test.npy", allow_pickle=True)
validation_data = np.load(r"datasets/validation.npy", allow_pickle=True)

columns = ["front X-Ray", "lateral X-Ray", "findings", "dec_ip", "dec_op", "image_features"]

train_data = pd.DataFrame(train_data, columns=columns)
test_data = pd.DataFrame(test_data, columns=columns)
validation_data = pd.DataFrame(validation_data, columns=columns)

# train_data.to_csv("datasets/train.csv")
# test_data.to_csv("datasets/test.csv")
# validation_data.to_csv("datasets/validation.csv")

training_img_features = np.vstack(train_data.image_features).astype(np.float)
validation_img_features = np.vstack(validation_data.image_features).astype(np.float)

tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(train_data['findings'])

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
vocabolary_length = len(tokenizer.word_index) + 1

train_decoder_ip = tokenizer.texts_to_sequences(train_data.dec_ip)
train_decoder_op = tokenizer.texts_to_sequences(train_data.dec_op)
val_decoder_ip = tokenizer.texts_to_sequences(validation_data.dec_ip)
val_decoder_op = tokenizer.texts_to_sequences(validation_data.dec_op)

max_len = 100
decoder_input_seq = pad_sequences(train_decoder_ip, maxlen=max_len, padding='post')
decoder_output_seq = pad_sequences(train_decoder_op, maxlen=max_len, padding='post')
Val_decoder_input_seq = pad_sequences(val_decoder_ip, maxlen=max_len, padding='post')
Val_decoder_output_seq = pad_sequences(val_decoder_op, maxlen=max_len, padding='post')

word_idx = {}
idx_word = {}
for key, value in tokenizer.word_index.items():
    word_idx[key] = value
    idx_word[value] = key

batch_size = 50
Buffer_size = 500

training_dataset = tf.data.Dataset.from_tensor_slices(((training_img_features, decoder_input_seq), decoder_output_seq))
training_dataset = training_dataset.shuffle(Buffer_size).batch(batch_size).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    ((validation_img_features, Val_decoder_input_seq), Val_decoder_output_seq))
validation_dataset = validation_dataset.shuffle(Buffer_size).batch(batch_size).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)


class Encoder_layer_stack(tf.keras.Model):

    def __init__(self, lstm_units):
        super().__init__()
        self.lstm_units = lstm_units
        self.dense = Dense(self.lstm_units, kernel_initializer="glorot_uniform", name='encoder_dense_layer')

    def initialize_states(self, batch_size):
        self.batch_size = batch_size
        self.enc_h = tf.zeros((self.batch_size, self.lstm_units))

        return self.enc_h

    def call(self, x):
        encoder_output = self.dense(x)
        return encoder_output


class Attention_layer(tf.keras.layers.Layer):

    def __init__(self, attention_units):
        super().__init__()

        self.attention_units = attention_units

        self.w1 = tf.keras.layers.Dense(self.attention_units, kernel_initializer="glorot_uniform",
                                        name='Concat_w1_Dense')
        self.w2 = tf.keras.layers.Dense(self.attention_units, kernel_initializer="glorot_uniform",
                                        name='Concat_w2_Dense')
        self.Concat_Dense = tf.keras.layers.Dense(1, kernel_initializer="glorot_uniform", name='Concat_Dense_layer')

    def call(self, x):
        self.decoder_hidden_state, self.encoder_output = x
        self.decoder_hidden_state = tf.expand_dims(self.decoder_hidden_state, axis=1)

        score = self.Concat_Dense(
            tf.nn.tanh(self.w1(self.decoder_hidden_state) + self.w2(self.encoder_output)))

        att_weights = tf.nn.softmax(score, axis=1)
        context_vector = att_weights * self.encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, att_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, attention_units):
        super().__init__()

        self.lstm_units = lstm_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.attention_units = attention_units

        self.dense = Dense(self.vocab_size, kernel_initializer="glorot_uniform", name='onestep_dense')
        self.attention = Attention_layer(self.attention_units)
        self.decoder_emb = Embedding(self.vocab_size, self.embedding_dim, trainable=True, name='Decoder_embedding')
        self.decoder_gru = GRU(self.lstm_units, return_state=True, return_sequences=True, name="Decoder_LSTM")

        self.dropout1 = Dropout(0.3, name='dropout1')
        self.dropout2 = Dropout(0.3, name='dropout2')
        self.dropout3 = Dropout(0.3, name='dropout3')

    @tf.function
    def call(self, x, training=None):
        self.input_to_decoder, self.encoder_output, self.state_h = x

        embedded_output = self.decoder_emb(self.input_to_decoder)
        embedded_output = self.dropout1(embedded_output)

        y = [self.state_h, self.encoder_output]
        context_vector, att_weights = self.attention(y)

        concated_decoder_input = tf.concat([tf.expand_dims(context_vector, 1), embedded_output], -1)
        concated_decoder_input = self.dropout2(concated_decoder_input)

        output_gru, hidden_state = self.decoder_gru(concated_decoder_input, initial_state=self.state_h)

        output_gru = tf.reshape(output_gru, (-1, output_gru.shape[2]))
        output_gru = self.dropout3(output_gru)

        output = self.dense(output_gru)

        return output, hidden_state, att_weights, context_vector


class Decoder_layer_stack(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, attention_units):
        super().__init__()

        self.lstm_units = lstm_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.attention_units = attention_units

        self.onestepdecoder = Decoder(self.vocab_size, self.embedding_dim, self.lstm_units, self.attention_units)

    @tf.function
    def call(self, x, training=None):
        self.input_to_decoder, self.encoder_output, self.decoder_hidden_state = x
        all_outputs = tf.TensorArray(tf.float32, size=self.input_to_decoder.shape[1], name='output_arrays')

        for timestep in tf.range(self.input_to_decoder.shape[1]):
            y = [self.input_to_decoder[:, timestep:timestep + 1], self.encoder_output, self.decoder_hidden_state]
            output, hidden_state, att_weights, context_vector = self.onestepdecoder(y)

            self.decoder_hidden_state = hidden_state
            all_outputs = all_outputs.write(timestep, output)

        all_outputs = tf.transpose(all_outputs.stack(), [1, 0, 2])

        return all_outputs


class Encoder_decoder_model(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, lstm_units, attention_units, batch_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.embedding_dim = embedding_dim
        self.attention_units = attention_units

        self.encoder = Encoder_layer_stack(self.lstm_units)
        self.decoder = Decoder_layer_stack(vocab_size, embedding_dim, lstm_units, attention_units)
        self.dense = Dense(self.vocab_size, kernel_initializer="glorot_uniform", name='enc_dec_dense')

    def call(self, data):
        self.inputs, self.outputs = data[0], data[1]

        self.encoder_hidden = self.encoder.initialize_states(self.batch_size)
        self.encoder_output = self.encoder(self.inputs)

        x = [self.outputs, self.encoder_output, self.encoder_hidden]
        output = self.decoder(x)

        return output


optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


lstm_units = 256
embedding_dim = 300
attention_units = 64
tf.keras.backend.clear_session()
Attention_model = Encoder_decoder_model(vocabolary_length, embedding_dim, lstm_units, attention_units, batch_size)
Attention_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=loss_function)
Attention_model.fit(training_dataset, validation_data=validation_dataset, epochs=1, shuffle=True)

Attention_model.load_weights(r"datasets/Attention_model_final.h5")


def load_image_to_chexnet(img):
    image = Image.open(img)
    op_img = np.asarray(image.convert("RGB"))
    op_img = np.asarray(op_img)
    op_img = preprocess_input(op_img)
    op_img = resize(op_img, (256, 256, 3))
    op_img = np.expand_dims(op_img, axis=0)
    op_img = np.asarray(op_img)

    return op_img


def preprocess_imgs(image1_path, image2_path):
    image_features = []
    for i in range(len(image1_path)):
        i1 = load_image_to_chexnet(image1_path)
        i2 = load_image_to_chexnet(image2_path)
        img1_features = model.predict(i1)
        img2_features = model.predict(i2)
        img1_features = np.vstack(img1_features).astype(np.float)
        img2_features = np.vstack(img2_features).astype(np.float)

        total_tensor = np.concatenate((img1_features, img2_features), axis=1)

    return total_tensor


def predict_report(image1, image2):
    img_tensor = preprocess_imgs(image1, image2)
    image_features = np.vstack(img_tensor).astype(np.float)

    result = ''
    initial_state = Attention_model.layers[0].initialize_states(1)
    sequences = [['<start>', initial_state, 0]]

    encoder_output = Attention_model.layers[0](image_features)
    decoder_hidden_state = initial_state

    max_len = 75
    beam_width = 3
    finished_seq = []

    for i in range(max_len):
        new_seq = []
        all_probable = []

        for seq, state, score in sequences:

            cur_vec = np.reshape(word_idx[seq.split(" ")[-1]], (1, 1))
            decoder_hidden_state = state
            x = [cur_vec, encoder_output, decoder_hidden_state]
            output, hidden_state, att_weights, context_vector = Attention_model.decoder.onestepdecoder(x)
            output = tf.nn.softmax(output)
            top_words = np.argsort(output).flatten()[-beam_width:]
            for index in top_words:
                predicted = [seq + ' ' + idx_word[index], hidden_state,
                             score - np.log(np.array(output).flatten()[index])]
                all_probable.append(predicted)

        sequences = sorted(all_probable, key=lambda l: l[2])[:beam_width]

        count = 0
        for seq, state, score in sequences:
            if seq.split(" ")[-1] == '<end>':
                score = score / len(seq)
                finished_seq.append([seq, state, score])
                count += 1
            else:
                new_seq.append([seq, state, score])

        sequences = new_seq
        beam_width = beam_width - count
        if not sequences:
            break
        else:
            continue

    if len(finished_seq) > 0:
        finished_seq = sorted(finished_seq, reverse=True, key=lambda l: l[2])
        sequences = finished_seq[-1]
        return sequences[0][8:-5]
    else:
        return new_seq[-1][0]


app = Flask(__name__)
run_with_ngrok(app)


@app.route('/')
def home():
    return flask.render_template('index.html')


@app.route('/submit', methods=['POST'])
def predict_caption():
    front_XRay = request.files['file_1']
    lateral_XRay = request.files['file_2']

    cwd = os.getcwd()
    front_XRay_path = os.path.join(cwd, 'static', secure_filename(front_XRay.filename))
    lateral_XRay_path = os.path.join(cwd + 'static' + secure_filename(lateral_XRay.filename))

    front_XRay.save(front_XRay_path)
    lateral_XRay.save(lateral_XRay_path)

    result = predict_report(front_XRay_path, lateral_XRay_path)
    print(front_XRay_path)
    print(type(front_XRay_path))
    return flask.render_template('index.html', prediction=result, front_XRay=front_XRay_path,
                                 lateral_XRay=lateral_XRay_path)


if __name__ == '__main__':
    app.run()
