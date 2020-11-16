from keras.layers import (Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Multiply, GlobalMaxPool1D, Dense, Dropout, Activation, Reshape, Input, Concatenate, Add, ZeroPadding1D)

from keras.regularizers import l2
from keras.models import Model
from layer import MusicSinc1D

def se_fn(x, amplifying_ratio):
    num_features = x.shape[-1].value
    x = GlobalAvgPool1D()(x)
    x = Reshape((1, num_features))(x)
    x = Dense(num_features * amplifying_ratio, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(num_features, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
    return x

def rese_block(x, num_features, weight_decay, amplifying_ratio):
    if num_features != x.shape[-1].value:
      shortcut = Conv1D(num_features, kernel_size=1, padding='same', use_bias=True,
                      kernel_regularizer=l2(weight_decay), kernel_initializer='glorot_uniform')(x)
      shortcut = BatchNormalization()(shortcut)
    else:
      shortcut = x
    x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
             kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
             kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    if amplifying_ratio > 0:
      x = Multiply()([x, se_fn(x, amplifying_ratio)])
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)
    return x


def music_sincnet(x, filter_size=2501, sr=22050, filter_num=256, amplifying_ratio=16, drop_rate=0.5, weight_decay=0., num_classes=50):
    
    # Input&Reshape
    x = Input(tensor=x)
    x = Reshape([-1, 1])(x)
    # MusicSinc
    x = ZeroPadding1D(padding=(filter_size-1)//2)(x)
    x = MusicSinc1D(filter_num, filter_size, sr)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Strided Conv
    num_features = int(filter_num//2)
    x = Conv1D(num_features, kernel_size=3, strides=3, padding='valid', use_bias=True,
             kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # rese-block
    layer_outputs = []
    for i in range(9):
        num_features *= 2 if (i == 2 or i == 8) else 1
        x = rese_block(x, num_features, weight_decay, amplifying_ratio)
        layer_outputs.append(x)

    x = Concatenate()([GlobalMaxPool1D()(output) for output in layer_outputs[-3:]])
    x = Dense(x.shape[-1].value, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if drop_rate > 0.:
        x = Dropout(drop_rate)(x)
    x = Dense(num_classes, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
    return x
