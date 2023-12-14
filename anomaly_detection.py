# 필요 라이브러리 호출
import os
import sys
import numpy as np
import pandas as pd
# we will only import certain module from those libraries
from mpl_toolkits.mplot3d import Axes3D
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
import tensorflow as tf
import logging
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


# anomalies.py (1)
class Anomaly(object):
 def __init__(self):
  pass

  ## 이상치_함수_1 : _deltas(self, errors, epsilon, mean, std)
  """평균과 표준 편차 차이를 계산
    delta_mean = mean(errors) - mean(epsilon 이하의 모든 오류)
    delta_std = std(errors) - std(epsilon 이하의 모든 오류)

    매개변수:
    errors (ndarray): 오류 배열.
    epsilon (ndarray): 임계값.
    mean (float): 오류의 평균.
    std (float): 오류의 표준 편차.

    Returns:
    float, float:
    * delta_mean.
    * delta_std.
  """
 def _deltas(self, errors, epsilon, mean, std):
   below = errors[errors <= epsilon]
   if not len(below):
     return 0, 0
   return mean - below.mean(), std - below.std()


  ## 이상치_함수_2 : _count_above(self, errors, epsilon)
 """epsilon 이상의 오류 및 연속된 시퀀스의 수를 세어줌
    연속된 시퀀스는 이동(shift)하고 원래 값이 true이며 변경이 있었던 위치의 수를 세어,
    해당 위치에서 시퀀스가 시작했음을 의미합니다.

    매개변수:
    errors (ndarray): 오류 배열.
    epsilon (ndarray): 임계값.

    Returns:
    int, int:
    * epsilon 이상의 오류의 수.
    * epsilon 이상의 연속된 시퀀스의 수.
 """
 def _count_above(self, errors, epsilon):
   above = errors > epsilon # 임계값을 초과하는 모든 오류 값 필터링
   total_above =len(errors[above]) # 임계값을 초과하는 오류 총 개수 계산
   
   # 연속된 시퀀스 계산
   above = pd.Series(above)
   shift = above.shift(1) # 시리즈를 한칸씩 이동 -> 이전 요소와 현재 요소를 비교
   change = above != shift # 현재 값과 이동된 값이 다른 경우를 찾아 change에 저장
   total_consecutive = sum(above & change)
   return total_above, total_consecutive


  ## 이상치_함수_3 : _z_cost(self, z, errors, mean, std):
 """z 값이 얼마나 나쁜지를 계산

    원 계산방식
    z = (delta_mean/mean) + (delta_std/std)
    ------------------------------------------------------

    epsilon 이상의 오류 수 + (epsilon 이상의 연속된 시퀀스 수)^2
    이는 `z`의 "우수성"을 계산하는 것으로, 값이 높을수록 `z`가 더 좋다는 것을 의미
    이 경우, 이 값을 반전하여 (음수로 만들어) 비용 함수로 변환하며, 나중에 scipy.fmin을 사용하여 최소화

    매개변수:
    z (ndarray): 비용 점수를 계산할 값.
    errors (ndarray): 오류 배열.
    mean (float): 오류의 평균.
    std (float): 오류의 표준 편차.
    Returns float: z의 비용.
 """
 def _z_cost(self, z, errors, mean, std):
   epsilon = mean + z * std
   delta_mean, delta_std =self._deltas(errors, epsilon, mean, std)
   above, consecutive =self._count_above(errors, epsilon)
   numerator =-(delta_mean / mean + delta_std / std)
   denominator = above + consecutive **2
   if denominator ==0:
     return np.inf
   return numerator / denominator


  ## 이상치_함수_4_1 : _find_threshold(self, errors, z_range):
 """이상적인 임계값 탐색
    이상적인 임계값은 z_cost 함수를 최소화
    scipy.fmin을 사용하여 최소값을 찾으며, z_range의 값을 시작점으로 사용합니다.

    매개변수:
    errors (ndarray): 오류 배열.
    z_range (list):
    scipy.fmin 함수의 시작점으로 선택되는 범위를 나타내는 두 값을 포함한 리스트.

    Returns: float: 계산된 임계값.
 """
 def _find_threshold(self, errors, z_range):
   mean = errors.mean()
   std = errors.std()
   min_z, max_z = z_range
   best_z = min_z
   best_cost = np.inf
   for z in range(min_z, max_z):
     best = fmin(self._z_cost, z, args=(errors, mean, std), full_output=True,
     disp=False)
     z, cost = best[0:2]
     if cost < best_cost:
       best_z = z[0]
   return mean + best_z * std

  ## 이상치_함수_4_2 : _fixed_threshold(self, errors, z_range):
 """임계값을 계산합니다.
    고정 임계값은 평균에서 k 표준 편차만큼 떨어진 값으로 정의됩니다.
    Args:
    errors (ndarray): 오류 배열.
    Returns:
    float: 계산된 임계값.
 """
 def _fixed_threshold(self, errors, k=3.0):
   mean = errors.mean()
   std = errors.std()
   return mean + k * std


  ## 이상치_함수_5_1 : _find_sequences(self, errors, epsilon, anomaly_padding)
 """epsilon 이상의 값을 갖는 시퀀스를 찾습니다.

    다음 단계를 따릅니다:
    * epsilon 이상의 값을 나타내는 부울 마스크를 생성
    * True 값 주변의 일정 범위의 오류를 True로 표시
    * 이 마스크를 한 칸 이동시켜 빈 갭을 False로 채dna
    * 이동된 마스크를 원래 마스크와 비교하여 변경사항이 있는지 확인
    * True에서 False로 변경된 지점을 시퀀스의 시작점
    * False에서 True로 변경된 지점을 시퀀스의 끝점

    매개변수:
    errors (ndarray): 오류 배열.
    epsilon (float): 임계값. epsilon 이상의 모든 오류는 이상으로 간주됩니다.
    anomaly_padding (int):
    찾은 이상 현상의 전후에 추가되는 오류의 수.

    Returns:
    ndarray, float:
    * 각 발견된 이상 시퀀스의 시작 및 끝을 포함하는 배열.
    * 이상으로 간주되지 않은 최대 오류 값.
 """
 def _find_sequences(self, errors, epsilon, anomaly_padding):
   above = pd.Series(errors > epsilon)
   index_above = np.argwhere(above.values)
   for idx in index_above.flatten():
     above[max(0, idx - anomaly_padding):min(idx + anomaly_padding +1,
     len(above))] =True
   shift = above.shift(1).fillna(False)
   change = above != shift
   if above.all():
     max_below =0
   else:
     max_below = max(errors[~above])
   index = above.index
   starts = index[above & change].tolist()
   ends = (index[~above & change]-1).tolist()
   if len(ends) ==len(starts)-1:
     ends.append(len(above)-1)
   return np.array([starts, ends]).T, max_below

  ## 이상치_함수_5_2 : _get_max_errors(self, errors, sequences, max_below)
 """각 이상 시퀀스에 대한 최대 오류를 가져옴
    또한 이상으로 간주되지 않은 최대 오류 값을 포함하는 행을 추가
    ``max_error`` 열을 포함한 테이블로 각 시퀀스의 최대 오류와 해당 시작 및 중지 인덱스가 최대 오류로 내림차순으로 정렬된 DataFrame입니다.

    매개변수:
    errors (ndarray): 오류 배열.
    sequences (ndarray): 이상 시퀀스의 시작 및 끝을 포함하는 배열.
    max_below (float): 이상으로 간주되지 않은 최대 오류 값.

    Returns:
    pandas.DataFrame: ``start``, ``stop``, 및 ``max_error`` 열을 포함하는 DataFrame 객체.
 """
 def _get_max_errors(self, errors, sequences, max_below):
    max_errors = [{
    'max_error': max_below,
    'start': -1,
    'stop': -1
    }]
    for sequence in sequences:
      start, stop = sequence
      sequence_errors = errors[start: stop +1]
      max_errors.append({
        'start': start,
        'stop': stop,
        'max_error': max(sequence_errors)
    })
    max_errors = pd.DataFrame(max_errors).sort_values('max_error', ascending=False)
    return max_errors.reset_index(drop=True)

  ## 이상치_함수_6 : _prune_anomalies(self, max_errors, min_percent)
 """거짓 양성을 줄이기 위해 이상을 가려냄

    단계:
    1) 각 값을 다음 값과 비교하기 위해 오류를 1단계 음수로 이동
    2) 비교하고 싶지 않은 마지막 행을 삭제
    3) 각 행에 대한 증가 비율을 계산
    4) ``min_percent`` 미만인 행을 탐색
    5) 그러한 행 중 가장 최근 행의 인덱스를 탐색
    6) 그 인덱스 위의 모든 시퀀스의 값을 가져옵니다.

    매개변수:
    max_errors (pandas.DataFrame) : ``start``, ``stop``, 및 ``max_error`` 열을 포함하는 DataFrame 객체.
    min_percent (float) : 이상은 서로와 윈도우 시퀀스의 최고 비이상 오류 간의 분리를 충족해야 하는 백분율.

    Returns:
    ndarray: 가려진 이상의 시작, 끝, 최대 오류를 포함하는 배열.
 """
 def _prune_anomalies(self, max_errors, min_percent):
    next_error = max_errors['max_error'].shift(-1).iloc[:-1]
    max_error = max_errors['max_error'].iloc[:-1]
    increase = (max_error-next_error) / max_error
    too_small = increase < min_percent
    if too_small.all():
      last_index =-1
    else:
      last_index = max_error[~too_small].index[-1]
    return max_errors[['start', 'stop', 'max_error']].iloc[0: last_index+1].values

  ## 이상치_함수_7 : _compute_scores(self, pruned_anomalies, errors, threshold, window_start):
 """이상의 점수를 계산
    시퀀스 내의 최대 오류에 비례하여 이상의 점수를 계산하고
    인덱스를 절대적으로 만들기 위해 window_start 타임스탬프를 추가

    매개변수:
    pruned_anomalies (ndarray): 창 내의 모든 이상을 포함하는 시작, 끝 및 최대 오류를 포함하는 이상 배열.
    errors (ndarray): 오류 배열.
    threshold (float): 임계값.
    window_start (int): 창 내의 첫 번째 오류 값의 인덱스.
    Returns:
    list: 각 이상에 대한 시작 인덱스, 끝 인덱스, 점수를 포함하는 이상 목록.
 """
 def _compute_scores(self, pruned_anomalies, errors, threshold, window_start):
    anomalies = list()
    denominator = errors.mean() + errors.std()
    for row in pruned_anomalies:
      max_error = row[2]
      score = (max_error-threshold) / denominator
      anomalies.append([row[0]+window_start, row[1]+window_start, score])
    return anomalies

  ## 이상치_함수_8 : _merge_sequences(self, sequences):
 """연속 및 겹치는 시퀀스를 병합
    시작, 끝, 점수 트리플의 목록을 반복하고 겹치거나 연속하는 시퀀스를 병합
    병합된 시퀀스의 점수는 해당 시퀀스의 길이로 가중 평균한 단일 점수

    매개변수:
    sequences (list): 각 이상에 대한 시작 인덱스, 끝 인덱스, 점수를 포함하는 이상 목록.

    Returns:
    ndarray: 병합 후 각 이상에 대한 시작 인덱스, 끝 인덱스, 점수를 포함하는 배열.
 """
 def _merge_sequences(self, sequences):
    if len(sequences) ==0:
      return np.array([])
    sorted_sequences = sorted(sequences, key=lambda entry: entry[0])
    new_sequences = [sorted_sequences[0]]
    score = [sorted_sequences[0][2]]
    weights = [sorted_sequences[0][1] - sorted_sequences[0][0]]
    for sequence in sorted_sequences[1:]:
      prev_sequence = new_sequences[-1]
      if sequence[0] <= prev_sequence[1] +1:
        score.append(sequence[2])
        weights.append(sequence[1] - sequence[0])
        weighted_average = np.average(score, weights=weights)
        new_sequences[-1] = (prev_sequence[0],
          max(prev_sequence[1],
          sequence[1]), weighted_average)
      else:
        score = [sequence[2]]
        weights = [sequence[1] - sequence[0]]
        new_sequences.append(sequence)
    return np.array(new_sequences)

  ## 이상치_함수_9 : _find_window_sequences(self, window, z_range, anomaly_padding, min_percent, window_start, fixed_threshold)
 """이상값을 탐색. window(창)의 임계값을 찾은 다음, 그 임계값을 초과하는 모든 시퀀스 탐색
    그 후, 시퀀스의 최대 오류(max_errors)를 구하여 이상을 판별하고,이상의 점수(_compute_scores()의 결과값 anomalies)를 계산

    매개변수:
    window (ndarray): 분석 중인 창의 오류 배열.
    z_range (list):
    동적 find_threshold 함수에 대한 시작점으로 선택되는 범위를 나타내는 두 값을 포함한 리스트.
    anomaly_padding (int):
    찾은 이상 현상의 전후에 추가되는 오류의 수.
    min_percent (float):
    이상은 서로와 윈도우 시퀀스의 최고 비이상 오류 간의 분리를 충족해야 하는 백분율.
    window_start (int): 창 내의 첫 번째 오류 값의 인덱스.
    fixed_threshold (bool): 고정 임계값을 사용할지 동적 임계값을 사용할지를 입력

    Returns:
    ndarray : 창에서 찾은 각 이상 시퀀스의 시작 인덱스, 끝 인덱스, 점수를 포함하는 배열.
 """
 def _find_window_sequences(self, window, z_range, anomaly_padding, min_percent, window_start, fixed_threshold):
    if fixed_threshold: threshold =self._fixed_threshold(window)
    else:
      threshold =self._find_threshold(window, z_range)
    window_sequences, max_below =self._find_sequences(window, threshold,
    anomaly_padding)
    max_errors =self._get_max_errors(window, window_sequences, max_below)
    pruned_anomalies =self._prune_anomalies(max_errors, min_percent)
    window_sequences =self._compute_scores(pruned_anomalies, window, threshold,
        window_start)
    return window_sequences

  ## 이상치_함수_10_1 : find_anomalies(self, errors, index, z_range=(0, 10), window_size=None,
    # window_size_portion=None, window_step_size=None,
    # window_step_size_portion=None, min_percent=0.1,
    # anomaly_padding=50, lower_threshold=False,
    # fixed_threshold=True)
 """이상한 오류 값의 시퀀스를 탐색
    1) 먼저 분석하려는 오류의 창을 정의
    2) 해당 창에서 이상한 시퀀스를 찾고 각 시퀀스에 해당하는 시작/중지 인덱스 쌍과 해당 점수를 저장
    3) 선택적으로 오류 시퀀스를 평균을 기준으로 뒤집어서 동일한 절차를 적용하여 비정상적으로 낮은 오류 시퀀스를 찾을 수 있습니다.
    4) 그런 다음 창을 이동시키고 프로시저를 반복합니다.
    5)마지막으로 겹치거나 연속하는 시퀀스를 결합합니다.

    매개변수:
    errors (ndarray): 오류 배열.
    index (ndarray): 오류의 인덱스 배열.
    z_range (list):
    선택 사항. scipy.fmin 함수에 대한 시작점으로 선택되는 범위를 나타내는 두 값을 포함한 리스트로, (0,10)이 기본값
    window_size (int):
    선택 사항. 임계값이 계산되는 창의 크기입니다. 제공되지 않으면 전체 오류 시퀀스에 대한 하나의 임계값을 탐색
    window_size_portion (float):
    선택 사항. 창의 크기를 오류 시퀀스의 일부로 지정. 제공되지 않으면 창 크기를 그대로 사용.
    window_step_size (int): 선택 사항. 새 창에 대한 임계값이 계산되기 전에 창을 이동하는 단계 수.
    window_step_size_portion (float):
    선택 사항. 단계 수를 창 크기의 일부로 지정. 제공되지 않으면 창 단계 크기를 그대로 사용
    min_percent (float):
    선택 사항. 이상은 서로와 창 시퀀스의 최고 비이상 오류 간의 분리를 충족해야 하는 백분율. 0.1이 기본값
    anomaly_padding (int):
    선택 사항. 찾은 이상 현상의 전후에 추가되는 오류의 수. 제공되지 않으면 50이 사용
    lower_threshold (bool):
    선택 사항. 비정상적으로 낮은 오류를 찾기 위해 하한 임계값을 적용할지 여부를 나타냄. 'False'가 기본값
    fixed_threshold (bool):
    선택 사항. 고정 된 임계값 또는 동적 임계값을 사용할지 여부를 나타냅니다.`False`가 기본값

    Returns:
    ndarray: 찾은 각 이상 시퀀스에 대한 시작 인덱스, 끝 인덱스, 점수를 포함하는 배열.
 """
 def find_anomalies(self, errors, index, z_range=(0, 10), window_size=None,
    window_size_portion=None, window_step_size=None,
    window_step_size_portion=None, min_percent=0.1,
    anomaly_padding=50, lower_threshold=False,
    fixed_threshold=False):
    window_size = window_size or len(errors)
    if window_size_portion:
      window_size = np.ceil(len(errors) * window_size_portion).astype('int')
    window_step_size = window_step_size or window_size
    if window_step_size_portion:
      window_step_size = np.ceil(window_size*window_step_size_portion).astype('int')
    window_start =0
    window_end =0
    sequences = list()
    # anomalies.py (10-3)
    while window_end <len(errors):
      window_end = window_start + window_size
      window = errors[window_start:window_end]
      window_sequences =self._find_window_sequences(window,
                z_range,
                anomaly_padding,
                min_percent,
                window_start,
                fixed_threshold)
      sequences.extend(window_sequences)
      if lower_threshold:
        # Flip errors sequence around mean
        mean = window.mean()
        inverted_window = mean - (window - mean)
        inverted_window_sequences=self._find_window_sequences(inverted_window,
                  z_range,
                  anomaly_padding,
                  min_percent,
                  window_start,
                  fixed_threshold)
        sequences.extend(inverted_window_sequences)
      window_start = window_start + window_step_size
    sequences =self._merge_sequences(sequences)
    anomalies = list()
    for start, stop, score in sequences:
      print("start", start)
      print("stop", stop)
      print("score", score)
      anomalies.append([index[int(start)], index[int(stop)], score])
    return anomalies
################################################################################
# find_anomalies 함수 수정
 # 1번 실험
#  def find_anomalies(self, errors, index, z_range=(0, 10), window_size=None,
#                    window_size_portion=0.1, window_step_size=None,
#                    window_step_size_portion=0.05, min_percent=0.1,
#                    anomaly_padding=50, lower_threshold=False,
#                    fixed_threshold=True):
 
#  # 2번 실험 더 넓은 z_range(XXXXXXXXXXXXXXXXXX)
#  # z_range를 줄임으로써, 이상치를 탐지하는 민감도를 낮춥니다.
#  # window_size_portion과 window_step_size_portion을 줄여 더 작은 윈도우에서 탐지를 수행합니다. (X)
# #  def find_anomalies(self, errors, index, z_range=(0, 5), window_size=None,
# #                    window_size_portion=0.1, window_step_size=None,
# #                    window_step_size_portion=0.01, min_percent=0.1,
# #                    anomaly_padding=50, lower_threshold=False,
# #                    fixed_threshold=True):

# # 3번 실험 더 작은 윈도우 크기와 스템 사이즈
# #  def find_anomalies(self, errors, index, z_range=(0, 10), window_size=None,
# #                    window_size_portion=0.02, window_step_size=None,
# #                    window_step_size_portion=0.005, min_percent=0.1,
# #                    anomaly_padding=50, lower_threshold=False,
# #                    fixed_threshold=True):
    
# # 4번 실험 # 더 높은 이상치 패딩 anomaly_padding을 늘려서, 각 이상치 주변의 데이터 포인트를 더 많이 포함시킵니다
# #  def find_anomalies(self, errors, index, z_range=(0, 10), window_size=None,
# #                    window_size_portion=0.1, window_step_size=None,
# #                    window_step_size_portion=0.05, min_percent=0.1,
# #                    anomaly_padding=100, lower_threshold=True,
# #                    fixed_threshold=True):

# # 5번 실험 하한 임계값 활성화 - 각 윈도우마다 동적으로 임계값을 설정
# #  def find_anomalies(self, errors, index, z_range=(0, 10), window_size=None,
# #                    window_size_portion=0.1, window_step_size=None,
# #                    window_step_size_portion=0.05, min_percent=0.1,
# #                    anomaly_padding=50, lower_threshold=False,
# #                    fixed_threshold=False):
    # 데이터 양에 따라 윈도우 크기와 스텝 사이즈를 동적으로 조정
    # window_size = int(len(errors) * window_size_portion) if window_size is None else window_size
    # window_step_size = int(window_size * window_step_size_portion) if window_step_size is None else window_step_size

    # window_start = 0
    # sequences = list()
    
    # while window_start + window_size <= len(errors):
    #     window_end = window_start + window_size
    #     window = errors[window_start:window_end]

    #     # 현재 윈도우에서 이상치 시퀀스 탐지
    #     window_sequences = self._find_window_sequences(window, z_range, anomaly_padding,
    #                                                    min_percent, window_start, fixed_threshold)
    #     sequences.extend(window_sequences)

    #     if lower_threshold:
    #         # 오류 시퀀스를 평균값 기준으로 반전시키고, 반전된 시퀀스에서도 이상치 탐지
    #         mean = window.mean()
    #         inverted_window = mean - (window - mean)
    #         inverted_window_sequences = self._find_window_sequences(inverted_window, z_range, anomaly_padding,
    #                                                                 min_percent, window_start, fixed_threshold)
    #         sequences.extend(inverted_window_sequences)

    #     window_start += window_step_size

    # # 중복되거나 겹치는 이상치 시퀀스 병합
    # sequences = self._merge_sequences(sequences)
    
    # # 최종 이상치 정보 생성
    # anomalies = []
    # for start, stop, score in sequences:
    #     anomalies.append([index[int(start)], index[int(stop)], score])

    # return anomalies




  ## 이상치_함수_11 : _compute_critic_score(self, critics, smooth_window):
 """이상 점수 배열을 계산

    매개변수:
    critics (ndarray): Critic 값.
    smooth_window (int): 부드러운 오류를 계산하는 데 적용되는 부드러운 창.

    Returns:
    ndarray: 이상 점수 배열.
 """
 def _compute_critic_score(self, critics, smooth_window):
    critics = np.asarray(critics)
    l_quantile = np.quantile(critics, 0.25)
    u_quantile = np.quantile(critics, 0.75)
    in_range = np.logical_and(critics >= l_quantile, critics <= u_quantile)
    critic_mean = np.mean(critics[in_range])
    critic_std = np.std(critics)
    z_scores = np.absolute((np.asarray(critics) - critic_mean) / critic_std) +1
    z_scores = pd.Series(z_scores).rolling(smooth_window, center=True,
        min_periods=smooth_window //2).mean().values
    return z_scores

  ## 이상치_함수_12 : _regression_errors(y, y_hat, smoothing_window=0.01, smooth=True)
 """예측과 기대 출력을 비교하여 오차의 배열을 계산. smooth = True이면 생성된 오차 배열에 EWMA를 적용
    매개변수:
    y (ndarray): 실제 값.
    y_hat (ndarray): 예측된 값.
    smoothing_window (float):
    선택 사항. 부드럽게 하기 위한 창의 크기로 y의 총 길이의 비율로 표현됩니다. 제공되지 않으면 0.01이 사용됩니다.
    smooth (bool):
    선택 사항. 반환된 오류가 EWMA로 부드럽게 되어야 하는지 여부를 나타냅니다. 제공되지 않으면 `True`가 사용됩니다.

    Returns:
    ndarray: 오류의 배열.
  """
 def _regression_errors(y, y_hat, smoothing_window=0.01, smooth=True):
    errors = np.abs(y-y_hat)[:, 0]
    if not smooth:
      return errors
    smoothing_window =int(smoothing_window*len(y))
    return pd.Series(errors).ewm(span=smoothing_window).mean().values

  ## 이상치_함수_13 : regression_errors(y, y_hat, smoothing_window=0.01, smooth=True)
 """예측과 기대 출력을 비교하여 오차의 배열을 계산

    매개변수:
    y (ndarray): 실제 값.
    y_hat (ndarray): 예측된 값.
    smoothing_window (float):
    선택 사항. 부드럽게 하기 위한 창의 크기로 y의 총 길이의 비율로 표현. 0.01이 기본값
    smooth (bool):
    선택 사항. 반환된 오류가 EWMA로 부드럽게 되어야 하는지 여부를 나타냄. True가 기본값

    Returns:
    ndarray: 오류의 배열.
 """
 def regression_errors(y, y_hat, smoothing_window=0.01, smooth=True):
    errors = np.abs(y - y_hat)[:, 0]
    if not smooth:
      return errors
    smoothing_window =int(smoothing_window *len(y))
    return pd.Series(errors).ewm(span=smoothing_window).mean().values

  ## 이상치_함수_14_1 : _point_wise_error(self, y, y_hat)
 """예측된 값과 기대 값 사이의 점별 오차를 계산
    Args:
    y (ndarray): 실제 값.
    y_hat (ndarray): 예측된 값.
    Returns:
    ndarray: 부드러운 점별 오차의 배열.
 """
 def _point_wise_error(self, y, y_hat):
    y_abs = abs(y-y_hat)
    y_abs_sum = np.sum(y_abs, axis=-1)
    return y_abs_sum

  ## 이상치_함수_14_2 : _area_error(self, y, y_hat, score_window=10)
 """예측된 값과 기대 값 사이의 영역 오차를 계산
    Args:
    y (ndarray): 실제 값.
    y_hat (ndarray): 예측된 값.
    score_window (int):
    선택 사항. 점수를 계산하는 데 사용되는 창의 크기. 10이 기본값

    Returns:
    ndarray: 영역 오차의 배열.
 """
 def _area_error(self, y, y_hat, score_window=10):
    smooth_y = pd.Series(y).rolling(score_window,
          center=True,
          min_periods=score_window //2).apply(integrate.trapz)
    smooth_y_hat = pd.Series(y_hat).rolling(score_window,
          center=True,
          min_periods=score_window //2).apply(integrate.trapz)
    errors = abs(smooth_y-smooth_y_hat)
    return errors

  ## 이상치_함수_15 : _dtw_error(self, y, y_hat, score_window=10)
 """예측된 값과 기대 값 사이의 DTW 오차를 계산합니다.
    Args:
    y (ndarray): 실제 값.
    y_hat (ndarray): 예측된 값.
    score_window (int):
    선택 사항. 점수를 계산하는 데 사용되는 창의 크기입니다.
    제공되지 않으면 10이 사용됩니다.
    Returns:
    ndarray: DTW 오차의 배열.
 """
 def _dtw_error(self, y, y_hat, score_window=10):
    length_dtw = (score_window //2) *2 +1
    half_length_dtw = length_dtw //2
    # add padding
    y_pad = np.pad(y,
          (half_length_dtw, half_length_dtw),
          'constant',
          constant_values=(0, 0))
    y_hat_pad = np.pad(y_hat,
          (half_length_dtw, half_length_dtw),
          'constant',
          constant_values=(0, 0))
    i =0
    similarity_dtw = list()
    while i <len(y)-length_dtw:
      true_data = y_pad[i:i+length_dtw]
      true_data = true_data.flatten()
      pred_data = y_hat_pad[i:i+length_dtw]
      pred_data = pred_data.flatten()
      dist = dtw(true_data, pred_data)
      similarity_dtw.append(dist)
      i +=1
    errors = ([0] * half_length_dtw + similarity_dtw + [0] * (len(y)-len(similarity_dtw) -
        half_length_dtw))
    return errors
  ## 이상치_함수_16 : _reconstruction_errors(self, y, y_hat, step_size=1,
      # score_window=10, smoothing_window=0.01,
      # smooth=True, rec_error_type='point')
 """재구성 오류 배열을 계산합니다.
    예상 값과 예측된 값 간의 불일치를 재구성 오류 유형에 따라 계산합니다.
    Args:
    y (ndarray): 실제 값.
    y_hat (ndarray): 예측된 값. 각 타임스탬프에는 여러 예측이 있습니다.
    step_size (int):
    선택 사항. 예측된 값의 창 간의 단계 수를 나타냅니다.
    제공되지 않으면 1이 사용됩니다.
    score_window (int):
    선택 사항. 점수를 계산하는 데 사용되는 창의 크기입니다.
    제공되지 않으면 10이 사용됩니다.
    smoothing_window (float or int):
    선택 사항. 부동 소수점인 경우 전체 길이의 비율로 표현된 부드러운 창의 크기입니다.
    제공되지 않으면 0.01이 사용됩니다.
    smooth (bool):
    선택 사항. 반환된 오류가 부드럽게 되어야 하는지 여부를 나타냅니다.
    제공되지 않으면 `True`가 사용됩니다.
    rec_error_type (str):
    선택 사항. 재구성 오류 유형 ``["point", "area", "dtw"]``입니다.
    제공되지 않으면 "point"가 사용됩니다.
    Returns:
    ndarray: 재구성 오류 배열.
 """
 def _reconstruction_errors(self, y, y_hat, step_size=1,
      score_window=10, smoothing_window=0.01,
      smooth=True, rec_error_type='point'):
    if isinstance(smoothing_window, float):
      smoothing_window = min(math.trunc(len(y) * smoothing_window), 200)
    true=[]
    for i in range(len(y)):
      true.append(y[i][0])
    for it in range(len(y[-1]) -1): # contents of the last window included
      true.append(y[-1][it+1])
  # anomalies.py (16-2)
    predictions = []
    predictions_vs = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] -1)
    for i in range(num_errors):
      intermediate = []
      for j in range(max(0, i - num_errors + pred_length), min(i+1, pred_length)):
        intermediate.append(y_hat[i-j, j])
      ave_p = []
      if intermediate:
        predictions.append(np.average(intermediate, axis=0))
        predictions_vs.append([[
            np.min(np.asarray(intermediate)),
            np.percentile(np.asarray(intermediate), 25),
            np.percentile(np.asarray(intermediate), 50),
            np.percentile(np.asarray(intermediate), 75),
            np.max(np.asarray(intermediate))
          ]])
    true = np.asarray(true)
    predictions = np.asarray(predictions)
    predictions_vs = np.asarray(predictions_vs)
    # Compute reconstruction errors
    if rec_error_type.lower() =="point":
      errors =self._point_wise_error(true, predictions)
    elif rec_error_type.lower() =="area":
      errors =self._area_error(true, predictions, score_window)
    elif rec_error_type.lower() =="dtw":
      errors =self._dtw_error(true, predictions, score_window)
    # Apply smoothing
    if smooth:
      errors = pd.Series(errors).rolling(smoothing_window,
                  center=True,
              min_periods=smoothing_window //2).mean().values
    return errors, predictions_vs

  ## 이상치_함수_17 : score_anomalies(self, y, y_hat, critic, index,
        # score_window=10, critic_smooth_window=None,
        # error_smooth_window=None, smooth=True,
        # rec_error_type="point", comb="mult", lambda_rec=0.5)
 def score_anomalies(self, y, y_hat, critic, index,
        score_window=10, critic_smooth_window=None,
        error_smooth_window=None, smooth=True,
        rec_error_type="point", comb="mult", lambda_rec=0.5):
    """이상 점수 배열을 계산합니다.
      이상 점수는 재구성 오류와 크리틱 점수를 결합하여 계산됩니다.
      Args:
      y (ndarray): 실제 값.
      y_hat (ndarray): 예측된 값. 각 타임스탬프에는 여러 예측이 있습니다.
      index (ndarray): 각 y에 대한 시간 인덱스 (창의 시작 위치)
      critic (ndarray): 크리틱 점수. 각 타임스탬프에는 여러 크리틱 점수가 있습니다.
      score_window (int):
      선택 사항. 점수를 계산하는 데 사용되는 창의 크기입니다.
      제공되지 않으면 10이 사용됩니다.
      critic_smooth_window (int):
      선택 사항. 크리틱에 적용되는 부드러운 창의 크기입니다.
      제공되지 않으면 200이 사용됩니다.
      error_smooth_window (int):
      선택 사항. 오류에 적용되는 부드러운 창의 크기입니다.
      제공되지 않으면 200이 사용됩니다.
      smooth (bool):
      선택 사항. 오류가 부드럽게 되어야 하는지 여부를 나타냅니다.
      제공되지 않으면 `True`가 사용됩니다.
      rec_error_type (str):
      선택 사항. 재구성 오류를 계산하는 방법입니다. `["point", "area", "dtw"]` 중 하나일 수 있습니다.
      제공되지 않으면 'point'가 사용됩니다.
      comb (str):
      선택 사항. 크리틱 및 재구성 오류를 결합하는 방법입니다. `["mult", "sum", "rec"]` 중 하나일 수 있습니다.
      제공되지 않으면 'mult'가 사용됩니다.
      lambda_rec (float):
      선택 사항. `comb="sum"`인 경우 점수를 결합하는 데 사용되는 람다 가중 합입니다.
      제공되지 않으면 0.5가 사용됩니다.
      Returns:
      ndarray: 이상 점수의 배열.
    """

 # anomalies.py (17-2)
    critic_smooth_window = critic_smooth_window or math.trunc(y.shape[0]*0.01)
    error_smooth_window = error_smooth_window or math.trunc(y.shape[0]*0.01)
    step_size =1 # expected to be 1
    true_index = list(index) # no offset
    true = []
    for i in range(len(y)):
      true.append(y[i][0])
    for it in range(len(y[-1]) -1): # contents of the last window included
      true.append(y[-1][it+1])
      true_index.append((index[-1] + it +1))# extend the index (by inluding the last window part)
    true_index = np.array(true_index)# in order to cover the whole sequence of data
    critic_extended = list()
    for c in critic:
      critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())
    critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))
    critic_kde_max = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0]-1)
   # anomalies.py (17-3)
    for i in range(num_errors):
      critic_intermediate = []
      for j in range(max(0, i - num_errors + pred_length), min(i+1, pred_length)):
        critic_intermediate.append(critic_extended[i - j, j])
      if len(critic_intermediate) >1:
        discr_intermediate = np.asarray(critic_intermediate)
        try:
          critic_kde_max.append(discr_intermediate[np.argmax(stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
        except np.linalg.LinAlgError:
          critic_kde_max.append(np.median(discr_intermediate))
      else:
        critic_kde_max.append(np.median(np.asarray(critic_intermediate)))
    # Compute critic scores
    critic_scores =self._compute_critic_score(critic_kde_max, critic_smooth_window)
    # Compute reconstruction scores
    rec_scores, predictions =self._reconstruction_errors(y, y_hat, step_size, score_window, error_smooth_window, smooth, rec_error_type)
    rec_scores = stats.zscore(rec_scores)
    rec_scores = np.clip(rec_scores, a_min=0, a_max=None) +1
    # Combine the two scores
    if comb =="mult":
      final_scores = np.multiply(critic_scores, rec_scores)
    elif comb =="sum":
      final_scores = (1 - lambda_rec) * (critic_scores -1) + lambda_rec * (rec_scores -1)
    elif comb =="rec":
      final_scores = rec_scores
    else:
      raise ValueError('Unknown combination specified {}, use "mult", "sum", or "rec" instead.'.format(comb))
    true = [[t] for t in true]
    return final_scores, true_index, true, predictions