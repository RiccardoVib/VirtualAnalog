import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Embedding, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import numpy as np
import matplotlib.pyplot as plt


class PositionalEncoding(object):
    def __init__(self, position, d):
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        self._encoding = np.concatenate([sines, cosines], axis=-1)
        self._encoding = self._encoding[np.newaxis, ...]

        def _get_angles(self, position, i, d):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d))
            return position * angle_rates

        def get_positional_encoding(self):
            return tf.cast(self._encoding, dtype=tf.float32)

class MaskHandler(object):
    def padding_mask(self, sequence):
        sequence = tf.cast(tf.math.equal(sequence, 0), tf.float32)
        return sequence[:, tf.newaxis, tf.newaxis, :]

    def look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

class PreProcessingLayer(Layer):
    def __init__(self, num_neurons, vocabular_size):
        super(PreProcessingLayer, self).__init__()

        # Initialize
        self.num_neurons = num_neurons

        # Add embedings and positional encoding
        #self.embedding = Embedding(vocabular_size, self.num_neurons)
        positional_encoding_handler = PositionalEncoding(vocabular_size, self.num_neurons)
        self.positional_encoding = positional_encoding.get_positional_encoding()

        # Add embedings and positional encoding
        self.dropout = Dropout(0.1)

    def call(self, sequence, training, mask):
        sequence_lenght = tf.shape(sequence)[1]
        #sequence = self.embedding(sequence)

        sequence *= tf.math.sqrt(tf.cast(self.num_neurons, tf.float32))
        sequence += self.positional_encoding[:, :sequence_lenght, :]
        sequence = self.dropout(sequence, training=training)

        return sequence


class ScaledDotProductAttentionLayer():
    def calculate_output_weights(self, q, k, v, mask):
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention = qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(weights, v)

        return output, weights


class MultiHeadAttentionLayer(Layer):
    def __init__(self, num_neurons, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.num_neurons = num_neurons
        self.depth = num_neurons // self.num_heads
        self.attention_layer = ScaledDotProductAttentionLayer()

        self.q_layer = Dense(num_neurons)
        self.k_layer = Dense(num_neurons)
        self.v_layer = Dense(num_neurons)

        self.linear_layer = Dense(num_neurons)

    def split(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # Run through linear layers
        q = self.q_layer(q)
        k = self.k_layer(k)
        v = self.v_layer(v)

        # Split the heads
        q = self.split(q, batch_size)
        k = self.split(k, batch_size)
        v = self.split(v, batch_size)

        # Run through attention
        attention_output, weights = self.attention_layer.calculate_output_weights(q, k, v, mask)

        # Prepare for the rest of processing
        output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.num_neurons))

        # Run through final linear layer
        output = self.linear_layer(concat_attention)

        return output, weights


def build_multi_head_attention_layers(num_neurons, num_heads):
    multi_head_attention_layer = MultiHeadAttentionLayer(num_neurons, num_heads)
    dropout = tf.keras.layers.Dropout(0.1)
    normalization = LayerNormalization(epsilon=1e-6)
    return multi_head_attention_layer, dropout, normalization


def build_feed_forward_layers(num_neurons, num_hidden_neurons):
    feed_forward_layer = tf.keras.Sequential()
    feed_forward_layer.add(Dense(num_hidden_neurons, activation='relu'))
    feed_forward_layer.add(Dense(num_neurons))

    dropout = Dropout(0.1)
    normalization = LayerNormalization(epsilon=1e-6)
    return feed_forward_layer, dropout, normalization


class EncoderLayer(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads):
        super(EncoderLayer, self).__init__()

        # Build multi head attention layer and necessary additional layers
        self.multi_head_attention_layer, self.attention_dropout, self.attention_normalization = \
            build_multi_head_attention_layers(num_neurons, num_heads)

        # Build feed-forward neural network and necessary additional layers
        self.feed_forward_layer, self.feed_forward_dropout, self.feed_forward_normalization = \
            build_feed_forward_layers(num_neurons, num_hidden_neurons)

    def call(self, sequence, training, mask):
        # Calculate attention output
        attnention_output, _ = self.multi_head_attention_layer(sequence, sequence, sequence, mask)
        attnention_output = self.attention_dropout(attnention_output, training=training)
        attnention_output = self.attention_normalization(sequence + attnention_output)

        # Calculate output of feed forward network
        output = self.feed_forward_layer(attnention_output)
        output = self.feed_forward_dropout(output, training=training)

        # Combine two outputs
        output = self.feed_forward_normalization(attnention_output + output)

        return output


class DecoderLayer(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads):
        super(DecoderLayer, self).__init__()

        # Build multi head attention layers and necessary additional layers
        self.multi_head_attention_layer1, self.attention_dropout1, self.attention_normalization1 = \
            build_multi_head_attention_layers(num_neurons, num_heads)

        self.multi_head_attention_layer2, self.attention_dropout2, self.attention_normalization2 = \
            build_multi_head_attention_layers(num_neurons, num_heads)

        # Build feed-forward neural network and necessary additional layers
        self.feed_forward_layer, self.feed_forward_dropout, self.feed_forward_normalization = \
            build_feed_forward_layers(num_neurons, num_hidden_neurons)

    def call(self, sequence, enconder_output, training, look_ahead_mask, padding_mask):
        attnention_output1, attnention_weights1 = self.multi_head_attention_layer1(sequence, sequence, sequence,
                                                                                   look_ahead_mask)
        attnention_output1 = self.attention_dropout1(attnention_output1, training=training)
        attnention_output1 = self.attention_normalization1(sequence + attnention_output1)

        attnention_output2, attnention_weights2 = self.multi_head_attention_layer2(enconder_output, enconder_output,
                                                                                   attnention_output1, padding_mask)
        attnention_output2 = self.attention_dropout1(attnention_output2, training=training)
        attnention_output2 = self.attention_normalization1(attnention_output1 + attnention_output2)

        output = self.feed_forward_layer(attnention_output2)
        output = self.feed_forward_dropout(output, training=training)
        output = self.feed_forward_normalization(attnention_output2 + output)

        return output, attnention_weights1, attnention_weights2


class Encoder(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads, vocabular_size, num_enc_layers=6):
        super(Encoder, self).__init__()

        self.num_enc_layers = num_enc_layers

        self.pre_processing_layer = PreProcessingLayer(num_neurons, vocabular_size)
        self.encoder_layers = [EncoderLayer(num_neurons, num_hidden_neurons, num_heads) for _ in range(num_enc_layers)]

    def call(self, sequence, training, mask):
        sequence = self.pre_processing_layer(sequence, training, mask)
        for i in range(self.num_enc_layers):
            sequence = self.encoder_layers[i](sequence, training, mask)

        return sequence


class Decoder(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads, vocabular_size, num_dec_layers=6):
        super(Decoder, self).__init__()

        self.num_dec_layers = num_dec_layers

        self.pre_processing_layer = PreProcessingLayer(num_neurons, vocabular_size)
        self.decoder_layers = [DecoderLayer(num_neurons, num_hidden_neurons, num_heads) for _ in range(num_dec_layers)]

    def call(self, sequence, enconder_output, training, look_ahead_mask, padding_mask):
        sequence = self.pre_processing_layer(sequence, training, mask)

        for i in range(self.num_dec_layers):
            sequence, attention_weights1, attention_weights2 = self.dec_layers[i](sequence, enconder_output, training,
                                                                                  look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_attention_weights1'.format(i + 1)] = attention_weights1
            attention_weights['decoder_layer{}_attention_weights2'.format(i + 1)] = attention_weights2

        return sequence, attention_weights



class Transformer(Model):
    def __init__(self, num_layers, num_neurons, num_hidden_neurons, num_heads, input_vocabular_size, target_vocabular_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_neurons, num_hidden_neurons, num_heads, input_vocabular_size, num_layers)
        self.decoder = Decoder(num_neurons, num_hidden_neurons, num_heads, target_vocabular_size, num_layers)
        self.linear_layer = Dense(target_vocabular_size)

    def call(self, transformer_input, tar, training, encoder_padding_mask, look_ahead_mask, decoder_padding_mask):
        encoder_output = self.encoder(transformer_input, training, encoder_padding_mask)
        decoder_output, attention_weights = self.decoder(tar, encoder_output, training, look_ahead_mask, decoder_padding_mask)
        output = self.linear_layer(decoder_output)

        return output, attention_weights


class Schedule(LearningRateSchedule):
    def __init__(self, num_neurons, warmup_steps=4000):
        super(Schedule, self).__init__()

        self.num_neurons = tf.cast(num_neurons, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.num_neurons) * tf.math.minimum(arg1, arg2)