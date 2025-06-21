import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class SAN(Layer):
    def __init__(self, seq_len, pred_len, period_len, enc_in, station_type='adaptive', **kwargs):
        super(SAN, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.channels = enc_in
        self.station_type = station_type
        self.epsilon = 1e-5

        self.model_mean = MLP(seq_len // period_len, pred_len, enc_in, period_len, mode='mean')
        self.model_std = MLP(seq_len // period_len, pred_len, enc_in, period_len, mode='std')

    def normalize(self, inputs):
        if self.station_type == 'adaptive':
            bs, length, dim = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
            num_periods = length // self.period_len
            truncated_length = num_periods * self.period_len
            inputs_truncated = inputs[:, :truncated_length, :]
            inputs_reshaped = tf.reshape(inputs_truncated, [bs, num_periods, self.period_len, dim])

            mean = tf.reduce_mean(inputs_reshaped, axis=-2, keepdims=True)
            std = tf.math.reduce_std(inputs_reshaped, axis=-2, keepdims=True)

            norm_inputs = (inputs_reshaped - mean) / (std + self.epsilon)
            norm_inputs = tf.reshape(norm_inputs, [bs, truncated_length, dim])
            if truncated_length < length:
                padding = [[0, 0], [0, length - truncated_length], [0, 0]]
                norm_inputs = tf.pad(norm_inputs, padding, mode='CONSTANT')

            mean_all = tf.reduce_mean(norm_inputs, axis=1, keepdims=True)
            mean_transformed = self.model_mean(tf.reshape(mean, [bs, -1, dim])) + mean_all
            std_input = tf.reduce_mean(std, axis=2, keepdims=False)
            std_transformed = self.model_std(std_input)

            outputs = tf.concat([mean_transformed, std_transformed], axis=-1)
            return norm_inputs, outputs[:, -self.pred_len:, :]
        else:
            return inputs, None

    def call(self, inputs, mode='n', station_pred=None):
        if mode == 'n':
            return self.normalize(inputs)
        elif mode == 'd':
            return self.denormalize(inputs, station_pred)

    def denormalize(self, inputs, station_pred):
        mean = station_pred[:, :, :self.channels]
        std = station_pred[:, :, self.channels:]
        return inputs * (std + self.epsilon) + mean

    def get_config(self):
        config = super(SAN, self).get_config()
        config.update({
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'period_len': self.period_len,
            'enc_in': self.channels,
            'station_type': self.station_type
        })
        return config

class MLP(tf.keras.layers.Layer):
    def __init__(self, seq_len, pred_len, enc_in, period_len, mode):
        super(MLP, self).__init__()
        self.channels = enc_in
        self.pred_len = pred_len
        self.mode = mode
        self.final_activation = tf.keras.layers.ReLU() if mode == 'std' else tf.keras.layers.Activation('linear')

        self.input_layer = Dense(512, activation="relu")
        self.output_layer = Dense(enc_in * pred_len)
        
    def call(self, x):
        x = self.input_layer(x)
        x = self.output_layer(x)
        bs = tf.shape(x)[0]
        x = tf.reshape(x, [bs, self.pred_len, self.channels])
        return self.final_activation(x)

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'enc_in': self.channels,
            'period_len': self.period_len,
            'mode': self.mode
        })
        return config