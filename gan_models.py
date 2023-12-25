import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import math
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from random import randrange
import argparse
import collections
import logging
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Flatten, Dense, Reshape, UpSampling1D, TimeDistributed
from tensorflow.keras.layers import Activation, Conv1D, LeakyReLU, Dropout, Add, Layer
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.optimizers import Adam
import pydot
import pydotplus
from pydotplus import graphviz
from scipy import stats
from scipy import integrate
from scipy.optimize import fmin
from pyts.metrics import dtw
from pandas.plotting import register_matplotlib_converters
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam


## HyperParameter
win_size = 100
features_dim = 3
feat_dim = features_dim
params = {}
params['plot_network']=True
params['epochs']=30
params['batch_size']=64
params['n_critic']=5
params['learning_rate']=0.00005
params['latent_dim']=20
params['shape'] = [win_size, features_dim]
params['encoder_input_shape'] = [win_size, features_dim]
params['encoder_reshape_shape'] = [20, 1]
params['generator_input_shape'] = [20, 1]
params['generator_reshape_shape'] = [win_size, 1]
params['critic_x_input_shape'] = [win_size, features_dim]
params['critic_z_input_shape'] = [20, 1]
print("win_size = %d, features_dim = %d " % (win_size, features_dim))

## wasserstein_loss 손실함수
def wasserstein_loss(y_true, y_pred):
  return K.mean(y_true*y_pred)

## RandomWeightedAverage 선형결합 클래스
class RandomWeightedAverage(Layer):
  def __init__(self, batch_size):
    """initialize Layer
    Args:
    batch_size: 64
    """
    super().__init__()
    self.batch_size = batch_size
  def call(self, inputs, **kwargs):
    """calculate random weighted average
    Args:
    inputs[0] x: original input
    inputs[1] x_: predicted input
    """
    alpha = K.random_uniform((self.batch_size, 1, 1))
    return (alpha * inputs[0]) + ((1-alpha) * inputs[1])

## Training parameter
batch_size = params['batch_size']
n_critics = params['n_critic']
epochs = params['epochs']

## layer parameter
shape = params['shape'] # [win_size = 10, features_dim = 3]
window_size = shape[0]
feat_dim = shape[1]
latent_dim = params['latent_dim'] # params['latent_dim']	= 20
encoder_input_shape = params['encoder_input_shape']
generator_input_shape = params['generator_input_shape']
critic_x_input_shape = params['critic_x_input_shape']
critic_z_input_shape = params['critic_z_input_shape']
encoder_reshape_shape = params['encoder_reshape_shape']
generator_reshape_shape = params['generator_reshape_shape']

### encoder, generator, critic_x, critic_z layer 생성 함수
## 1) encoder layer
def build_encoder_layer(input_shape, encoder_reshape_shape):
    x = Input(shape=input_shape)
    model = tf.keras.models.Sequential([
        Bidirectional(LSTM(units=win_size, return_sequences=True)),
        Flatten(),
        Dense(20), # 20 = self.critic_z_input_shape[0]
        Reshape(target_shape=encoder_reshape_shape)]) # (20, 1)
    return Model(x, model(x)) # 입력값, 출력값을 사용하여 keras의 "Model" 생성


## 2) generator layer
def build_generator_layer(input_shape, generator_reshape_shape):
  # input_shape = (20, 1) / generator_reshape_shape = (50, 1)
  x = Input(shape=input_shape)
  model = tf.keras.models.Sequential([
    Flatten(),
    Dense(win_size), # 50 originally
    Reshape(target_shape=generator_reshape_shape), # (50, 1)
    Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat'),
    Dropout(rate=0.2),
    #UpSampling1D(size=2),
    UpSampling1D(size=1),
    Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat'),
    Dropout(rate=0.2),
    TimeDistributed(Dense(features_dim)), # features_dim >=1, multiple features
    Activation(activation='tanh')]) # ((None, 10, features_dim))
  return Model(x, model(x))


## 3) build_critic_x_layer
# k_size 설정
if win_size >= 30:
  k_size = 5
else:
  k_size = 2

def build_critic_x_layer(input_shape):
  x = Input(shape=input_shape)
  model = tf.keras.models.Sequential([
    Conv1D(filters=64, kernel_size=k_size),
    LeakyReLU(alpha=0.2),
    Dropout(rate=0.25),
    Conv1D(filters=64, kernel_size=k_size),
    LeakyReLU(alpha=0.2),
    Dropout(rate=0.25),
    Conv1D(filters=64, kernel_size=k_size),
    LeakyReLU(alpha=0.2),
    Dropout(rate=0.25),
    Conv1D(filters=64, kernel_size=k_size),
    LeakyReLU(alpha=0.2),
    Dropout(rate=0.25),
    Flatten(),
    Dense(units=1)])
  return Model(x, model(x))

## 4) build_critic_z_layer
def build_critic_z_layer(input_shape):
    x = Input(shape=input_shape)
    model = tf.keras.models.Sequential([
      Flatten(),
      Dense(units=100),
      LeakyReLU(alpha=0.2),
      Dropout(rate=0.2),
      Dense(units=100),
      LeakyReLU(alpha=0.2),
      Dropout(rate=0.2),
      Dense(units=1)])
    return Model(x, model(x))


### 모델 생성하기
## layer 생성하는 함수를 활용하여 encoder, generator, critic_x, critic_z 생성
learning_rate = 0.0005
# encoder_input_shape = [20, 1], encoder_reshape_shape = [20,1]
encoder = build_encoder_layer(input_shape=encoder_input_shape,
encoder_reshape_shape=encoder_reshape_shape)

# generater_input_shape = [20,1], generator_reshape_shape = [10,1]
generator = build_generator_layer(input_shape=generator_input_shape, generator_reshape_shape=generator_reshape_shape)

# critic_x_input_shape = [10, 3]
critic_x = build_critic_x_layer(input_shape=critic_x_input_shape)

# critic_z_input_shape = [20, 1]
critic_z = build_critic_z_layer(input_shape=critic_z_input_shape)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate) #레거시 옵티마이저 사용으로 변경

## 생성한 4가지 모델 layer로 critic_x_model, critic_z_model, interpolated_x, interpolated_z
z = Input(shape=(latent_dim, 1)) # (20, 1)
x = Input(shape=shape) # (10,3)
x_ = generator(z) # build_generator_layer(20,1) -> return (10,3) shape의 Generator(Decoding) Data
z_ = encoder(x) # build_encoder_layer(10,3) -> return (20,1) shape의 Encoding Data

fake_x = critic_x(x_) # (None,1) shape 가짜 데이터
valid_x = critic_x(x) # (None,1) shape 진짜 데이터

# fake_x, valid_x 데이터 선형 결합
interpolated_x = RandomWeightedAverage(batch_size)([x, x_])
critic_x_model = Model(inputs=[x, z], outputs=[valid_x, fake_x, interpolated_x])

fake_z = critic_z(z_) # Decoding 된 (None,1) data -> critic_z_layer
valid_z = critic_z(z) # Encoding 된 (None,1) data -> shape critic_z_layer

# fake_z, valid_z 데이터 선형 결합
interpolated_z = RandomWeightedAverage(batch_size)([z, z_])
critic_z_model = Model(inputs=[x, z], outputs=[valid_z, fake_z, interpolated_z])
z_gen = Input(shape=(latent_dim, 1))
x_gen_ = generator(z_gen)
x_gen = Input(shape=shape)
z_gen_ = encoder(x_gen)
x_gen_rec = generator(z_gen_)
fake_gen_x = critic_x(x_gen_)
fake_gen_z = critic_z(z_gen_)
encoder_generator_model = Model([x_gen, z_gen], [fake_gen_x, fake_gen_z, x_gen_rec])

## 가중치 모델 적용을 위한 경로지정
critic_x_model.load_weights('weight1_critic_x_model.h5')
critic_z_model.load_weights('weight2_critic_z_model.h5')
encoder_generator_model.load_weights('weight3_encoder_generator_model.h5')

### 생성한 critic_x, critic_z, enc_gen layer의 loss를 계산하는 함수
##1)  critic_x_train_on_batch
@tf.function
def critic_x_train_on_batch(x, z, valid, fake, delta):
  with tf.GradientTape() as tape:
    (valid_x, fake_x, interpolated) = critic_x_model(inputs=[x, z], training=True)
    with tf.GradientTape() as gp_tape:
      gp_tape.watch(interpolated)
      pred = critic_x(interpolated, training=True)
    grads = gp_tape.gradient(pred, interpolated)[0]
    grads = tf.square(grads)
    ddx = tf.sqrt(1e-8 + tf.reduce_sum(grads, axis=np.arange(1, len(grads.shape))))
    gp_loss = tf.reduce_mean((ddx-1.0) **2)
    loss = tf.reduce_mean(wasserstein_loss(valid, valid_x))
    loss += tf.reduce_mean(wasserstein_loss(fake, fake_x))
    loss += gp_loss*10.0
  gradients = tape.gradient(loss, critic_x_model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, critic_x_model.trainable_weights))
  return loss

## 2) critic_z_train_on_batch
@tf.function
def critic_z_train_on_batch(x, z, valid, fake, delta):
  with tf.GradientTape() as tape:
    (valid_z, fake_z, interpolated) = critic_z_model(inputs=[x, z], training=True)
    with tf.GradientTape() as gp_tape:
      gp_tape.watch(interpolated)
      pred = critic_z(interpolated, training=True)
    grads = gp_tape.gradient(pred, interpolated)[0]
    grads = tf.square(grads)
    ddx = tf.sqrt(1e-8 + tf.reduce_sum(grads, axis=np.arange(1, len(grads.shape))))
    gp_loss = tf.reduce_mean((ddx -1.0) **2)
    loss = tf.reduce_mean(wasserstein_loss(valid, valid_z))
    loss += tf.reduce_mean(wasserstein_loss(fake, fake_z))
    loss += gp_loss*10.0
  gradients = tape.gradient(loss, critic_z_model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, critic_z_model.trainable_weights))
  return loss

## 3) enc_gen_train_on_batch
@tf.function
def enc_gen_train_on_batch(x, z, valid):
  with tf.GradientTape() as tape:
    (fake_gen_x, fake_gen_z, x_gen_rec) = encoder_generator_model(inputs=[x, z], training=True)
    x = tf.squeeze(x)
    x_gen_rec = tf.squeeze(x_gen_rec)
    loss = tf.reduce_mean(wasserstein_loss(valid, fake_gen_x))
    loss += tf.reduce_mean(wasserstein_loss(valid, fake_gen_z))
    loss += tf.keras.losses.MSE(x, x_gen_rec)*10
    loss = tf.reduce_mean(loss)
  gradients = tape.gradient(loss, encoder_generator_model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, encoder_generator_model.trainable_weights))
  return loss

## 데이터 예측함수 predict
def predict(X):
  X = X.reshape((-1, shape[0], feat_dim)) # feat_dim : feature dimension (modified from 1: yeesj)
  z_ = encoder.predict(X)
  y_hat = generator.predict(z_)
  critic = critic_x.predict(X)
  return y_hat, critic
