import tensorflow as tf
from tensorflow.keras.layers import Layer

class DAIN(Layer):
    def __init__(self, num_features, eps=1e-6, momentum=0.1, affine=True, **kwargs):
        super(DAIN, self).__init__(**kwargs)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

    def build(self, input_shape):
        if self.affine:
            self.weight = self.add_weight(
                name='weight',
                shape=(1, 1, self.num_features),
                initializer='ones',
                trainable=True
            )
            self.bias = self.add_weight(
                name='bias',
                shape=(1, 1, self.num_features),
                initializer='zeros',
                trainable=True
            )
        
        self.mean_in = self.add_weight(
            name='mean_in',
            shape=(1, 1, self.num_features),
            initializer='zeros',
            trainable=False
        )
        self.var_in = self.add_weight(
            name='var_in',
            shape=(1, 1, self.num_features),
            initializer='ones',
            trainable=False
        )

        super(DAIN, self).build(input_shape)

    def call(self, x, training=None):
        if training:
            # Calculate instance-level mean and variance
            mean_in = tf.reduce_mean(x, axis=[1], keepdims=True)
            var_in = tf.reduce_mean(tf.square(x - mean_in), axis=[1], keepdims=True)
            
            # Update moving averages
            self.mean_in.assign(self.momentum * self.mean_in + (1 - self.momentum) * tf.reduce_mean(mean_in, axis=0, keepdims=True))
            self.var_in.assign(self.momentum * self.var_in + (1 - self.momentum) * tf.reduce_mean(var_in, axis=0, keepdims=True))
        else:
            mean_in = self.mean_in
            var_in = self.var_in

        # Normalize
        x_normalized = (x - mean_in) / tf.sqrt(var_in + self.eps)
        
        if self.affine:
            x_normalized = self.weight * x_normalized + self.bias
        
        return x_normalized

    def get_config(self):
        config = super(DAIN, self).get_config()
        config.update({
            'num_features': self.num_features,
            'eps': self.eps,
            'momentum': self.momentum,
            'affine': self.affine
        })
        return config