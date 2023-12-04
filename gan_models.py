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


## 0) HyperParameter
win_size = 10
features_dim = 3

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