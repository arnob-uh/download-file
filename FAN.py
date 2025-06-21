import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.models import Model
import numpy as np

class MLPfreq(Model):
    def __init__(self, seq_len, enc_in):
        super(MLPfreq, self).__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in  # Output shape should match enc_in (not pred_len)

        self.dense_freq = Dense(64, activation='relu')
        self.dense_all = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(enc_in)  # Ensure output matches norm_input's shape
        ])

    def call(self, main_freq, x):
        processed_freq = self.dense_freq(main_freq)
        combined_input = tf.concat([processed_freq, x], axis=-1)
        return self.dense_all(combined_input)


class FAN(Layer):
    def __init__(self, seq_len, enc_in, freq_topk=20, rfft=True, **kwargs):
        super(FAN, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.freq_topk = freq_topk
        self.rfft = rfft
        self.mlp_freq = MLPfreq(seq_len, enc_in)  # Output shape matches enc_in

    def call(self, inputs):
        norm_input, x_filtered = self.main_freq_part(inputs, self.freq_topk, self.rfft)
        if norm_input.shape[-1] == 0:
            norm_input = tf.zeros_like(inputs)
        processed_main_freq = self.mlp_freq(x_filtered, inputs)
        return norm_input + processed_main_freq  # Now both have the same shape

    def main_freq_part(self, x, k, rfft=True):
        if rfft:
            xf = tf.signal.rfft(x)
        else:
            xf = tf.signal.fft(tf.cast(x, tf.complex64))

        abs_xf = tf.abs(xf)

        # Ensure `k` does not exceed available frequency components
        max_k = tf.shape(abs_xf)[-1]
        k = tf.minimum(k, max_k)

        top_k_values, indices = tf.nn.top_k(abs_xf, k=k)

        batch_size = tf.shape(indices)[0]
        seq_len = tf.shape(indices)[1]

        batch_indices = tf.range(batch_size)
        seq_indices = tf.range(seq_len)

        batch_indices = tf.reshape(batch_indices, [-1, 1, 1])
        seq_indices = tf.reshape(seq_indices, [1, -1, 1])

        batch_indices = tf.broadcast_to(batch_indices, tf.shape(indices))
        seq_indices = tf.broadcast_to(seq_indices, tf.shape(indices))

        full_indices = tf.stack([batch_indices, seq_indices, indices], axis=-1)

        mask = tf.zeros_like(xf, dtype=tf.complex64)
        updates = tf.ones_like(top_k_values, dtype=tf.complex64)

        mask = tf.tensor_scatter_nd_update(mask, full_indices, updates)
        
        xf_filtered = xf * mask

        if rfft:
            x_filtered = tf.signal.irfft(xf_filtered)
        else:
            x_filtered = tf.signal.ifft(xf_filtered)

        norm_input = x - x_filtered
        return norm_input, x_filtered

    def get_config(self):
        config = super(FAN, self).get_config()
        config.update({
            "seq_len": self.seq_len,
            "enc_in": self.enc_in,
            "freq_topk": self.freq_topk,
            "rfft": self.rfft
        })
        return config