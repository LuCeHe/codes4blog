
import numpy as np
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.layers import *
from keras.models import Model

import tensorflow as tf

# https://github.com/YerevaNN/R-NET-in-Keras/blob/master/layers/Argmax.py
class Argmax(Layer):

    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        return K.argmax(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        del input_shape[self.axis]
        return tuple(input_shape)

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Argmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
def test_Argmax():
    
    print("""
          Test ArgMax Layer
          
          """)

    tokens_input = Input((None,))
    embed = Embedding(10, 3)(tokens_input)
    lstm = LSTM(20, return_sequences=True)(embed)
    softmax = TimeDistributed(Activation('softmax'))(lstm)
    token = Argmax()(softmax)
    model = Model(tokens_input, token)
    
    example_tokens = np.array([[1, 2, 7],
                               [3, 9, 6]])
    
    prediction = model.predict(example_tokens)
    print(prediction)
    
    weights = model.trainable_weights  # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g) 

        
class DifferentiableArgmax(Layer):

    def __init__(self):
        pass
    
    def __call__(self, inputs):
        
        # if it doesnt sum to one: normalize

        def prob2oneHot(x):
            # len should be slightly larger than the length of x
            len = 3
            a = K.pow(len*x, 100)
            sum_a = K.sum(a, axis=-1)
            sum_a = tf.expand_dims(sum_a, axis=1)
            onehot = tf.divide(a, sum_a)
            
            return onehot
            
        onehot = Lambda(prob2oneHot)(inputs)
        onehot = Lambda(prob2oneHot)(onehot)
        onehot = Lambda(prob2oneHot)(onehot)
        
        def onehot2token(x):
            cumsum = tf.cumsum(onehot, axis = -1, exclusive = True, reverse = True)
            rounding = 2*(K.clip(cumsum, min_value = .5, max_value = 1) - .5)
            token = tf.reduce_sum(rounding, axis = -1)
            return token
        
        token = Lambda(onehot2token)(onehot)
        return [inputs, token]


def test_Dargmax():
    print("""
          Test Differentiable Argmax Layer
          
          """)
    
    tokens_input = Input((None,))
    embed = Embedding(10, 3)(tokens_input)
    lstm = LSTM(20, return_sequences=False)(embed)
    softmax = Dense(3, activation='softmax')(lstm)  # TimeDistributed(Activation('softmax'))(lstm)
    token = DifferentiableArgmax()(softmax)
    model = Model(tokens_input, token)
    
    example_tokens = np.array([[1, 2, 7],
                               [3, 9, 6]])
    
    prediction = model.predict(example_tokens)
    print(prediction)
    
    weights = model.trainable_weights  # weight tensors    
    grad = tf.gradients(xs=weights, ys=model.output)
    
    print('')
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g) 

    
if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(4)


    #test_Argmax()
    test_Dargmax()
    
