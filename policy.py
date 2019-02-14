import tensorflow as tf

class NN:
    def __init__(self, input, output):
        init = tf.contrib.layers.xavier_initializer
        self.hidden_1 = tf.layers.Dense(units=32, activation=tf.nn.relu, bias_initializer=init())
        self.hidden_2 = tf.layers.Dense(units=32, activation=tf.nn.relu, bias_initializer=init())
        self.output = tf.layers.Dense(units=output, activation=tf.nn.relu, bias_initializer=init())
    
    def __call__(self, x):
        return self.output(
            self.hidden_2(
                self.hidden_1(x)
            )
        )

    def get_trainable(self):
        weights = []
        for layer in [self.hidden_1, self.hidden_2, self.output]:
            weights.extend(layer.trainable_variables)
        return weights
