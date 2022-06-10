
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
def get_variable(init_type='xavier', shape=None, name=None, minval=-0.001, maxval=0.001, mean=0,
                 stddev=0.001, dtype='float32'):
    if type(init_type) is str:
        init_type = init_type.lower()
    if init_type == 'tnormal':
        return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'uniform':
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'normal':
        return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'xavier':
        maxval = np.sqrt(6. / np.sum(shape))
        minval = -maxval
        print(name, 'initialized from:', minval, maxval)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'xavier_out':
        maxval = np.sqrt(3. / shape[1])
        minval = -maxval
        print(name, 'initialized from:', minval, maxval)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'xavier_in':
        maxval = np.sqrt(3. / shape[0])
        minval = -maxval
        print(name, 'initialized from:', minval, maxval)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'zero':
        return tf.Variable(tf.zeros(shape=shape, dtype=dtype), name=name)
    elif init_type == 'one':
        return tf.Variable(tf.ones(shape=shape, dtype=dtype), name=name)
    elif init_type == 'identity' and len(shape) == 2 and shape[0] == shape[1]:
        return tf.Variable(tf.diag(tf.ones(shape=shape[0], dtype=dtype)), name=name)
    elif 'int' in init_type.__class__.__name__ or 'float' in init_type.__class__.__name__:
        return tf.Variable(tf.ones(shape=shape, dtype=dtype) * init_type, name=name)