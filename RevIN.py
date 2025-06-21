import tensorflow as tf
from tensorflow.keras.layers import Layer

class RevIN(Layer):
    def __init__(self, num_features: int, mode='norm', eps=1e-5, affine=True, **kwargs):
        super(RevIN, self).__init__(**kwargs)
        self.num_features = num_features
        self.mode = mode
        self.eps = eps
        self.affine = affine
        
    def build(self, input_shape):
        self.actual_features = input_shape[-1]
        
        if self.affine:
            self.affine_weight = self.add_weight(
                shape=(self.actual_features,),
                initializer='ones',
                trainable=True,
                name='affine_weight'
            )
            self.affine_bias = self.add_weight(
                shape=(self.actual_features,),
                initializer='zeros',
                trainable=True,
                name='affine_bias'
            )
        super(RevIN, self).build(input_shape)

    def _get_statistics(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        variance = tf.math.reduce_variance(x, axis=1, keepdims=True)
        stdev = tf.sqrt(variance + self.eps)
        return mean, stdev

    def _normalize(self, x):
        mean, stdev = self._get_statistics(x)
        x = (x - mean) / stdev
        if self.affine:
            weight = tf.reshape(self.affine_weight, (1, 1, self.actual_features))
            bias = tf.reshape(self.affine_bias, (1, 1, self.actual_features))
            x = x * weight + bias
        return x, mean, stdev

    def _denormalize(self, x, mean, stdev):
        if self.affine:
            weight = tf.reshape(self.affine_weight, (1, 1, self.actual_features))
            bias = tf.reshape(self.affine_bias, (1, 1, self.actual_features))
            x = (x - bias) / (weight + self.eps)
        x = x * stdev + mean
        return x

    def call(self, inputs):
        if self.mode == 'norm':
            x, mean, stdev = self._normalize(inputs)
            return x
        elif self.mode == 'denorm':
            mean, stdev = self._get_statistics(inputs)
            return self._denormalize(inputs, mean, stdev)
        else:
            raise ValueError(f"Mode {self.mode} not recognized. Use 'norm' or 'denorm'.")

    def get_config(self):
        config = super(RevIN, self).get_config()
        config.update({
            'num_features': self.num_features,
            'mode': self.mode,
            'eps': self.eps,
            'affine': self.affine
        })
        return config