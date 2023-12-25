import pandas as pd
import numpy as np
from gan_models import *
from flask_socketio import SocketIO
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from anomaly_detection import Anomaly
# from anomaly_detection import _fixed_threshold
from flask import Flask
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import os


app = Flask(__name__)
socketio = SocketIO(app)
anomaly = Anomaly()

## 데이터 전처리 함수 1)
# 차분함수 정의 - 데이터프레임이 주어지면 차이를 구하고, 평활화를 추가하여, 지정된 대로 지연되도록 사전 처리
def diff_smooth_df(df, lags_n, diffs_n, smooth_n, diffs_abs=False, abs_features=False):
  if diffs_n >=1:
    # 데이터 차이 계산
    df = df.diff(diffs_n).dropna()

    # diffs_abs == True 일 때 절대값으로 변경
    if diffs_abs ==True:
      df = abs(df)

  if smooth_n >=2:
    # 데이터 평탄화를 위해 이동평균(rolling average)를 적용
    df = df.rolling(smooth_n).mean().dropna()

  if lags_n >=1:
    # 각 차원에 대해 해당 차원에 대한 차분 및 평활화 값의 각 lags_n 지연에 대해 새 열을 추가
    df_columns_new =[f'{col}_lag{n}' for n in range(lags_n+1) for col in df.columns]
    df = pd.concat([df.shift(n) for n in range(lags_n+1)], axis =1).dropna()
    df.columns = df_columns_new

  # feature_vector의 명확성을 위해 지연된 값(lagged_values) 열을 정렬
  df = df.reindex(sorted(df.columns), axis =1)

  # abs_features==True 인 경우 dataframe의 모든 값 절대값으로 반환
  if abs_features ==True:
    df = abs(df)
  return df

## 데이터 전처리 함수 1-2) PCA 분석
def do_pca(data_1, features_dim):
    # PCA 모델 생성
    pca = PCA(n_components=features_dim)
    
    # PCA를 통해 데이터 차원 축소
    data = pca.fit_transform(data_1)

    # 새로운 DataFrame을 만들기 위한 리스트 초기화
    df_1 = []
    
    # 각 행에 대해 정수 인덱스와 PCA 결과 값을 리스트에 추가
    for i in range(len(data)):
        row = [i + 1]  # 정수 인덱스 값 부여
        for jj in range(features_dim):
            row.append(data[i][jj])
        df_1.append(row)

    # 새로운 DataFrame 생성
    df = pd.DataFrame(df_1)
    
    # 컬럼 이름 설정
    columns_new = ['date']
    for i in range(1, features_dim + 1):
        pca_i = 'PCA_%s' % str(i)
        columns_new.append(pca_i)

    df.columns = columns_new

    return df

## 데이터 전처리 함수 2) 시간 집합화 함수
def time_segments_aggregate(X, interval, time_column, method=['mean']):
 if isinstance(X, np.ndarray):
    X = pd.DataFrame(X)
 X = X.sort_values(time_column).set_index(time_column)
 if isinstance(method, str):
    method = [method]
 start_ts = X.index.values[0]
 max_ts = X.index.values[-1]
 values = list()
 index = list()
 while start_ts <= max_ts:
    end_ts = start_ts + interval
    subset = X.loc[start_ts:end_ts-1]
    aggregated = [
      getattr(subset, agg)(skipna=True).values
      for agg in method
    ]
    values.append(np.concatenate(aggregated))
    index.append(start_ts)
    start_ts = end_ts
 return np.asarray(values), np.asarray(index)


## 데이터 전처리 함수 3) Gan 모델의 input_shape에 맞게 (10,3) 차원으로 데이터 묶음 함수 처리
def rolling_window_sequences(X, index, window_size, target_size, step_size,
 target_column, drop=None, drop_windows=False):
    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()
    target = X[:, target_column]
    if drop_windows:
      if hasattr(drop, '__len__') and (not isinstance(drop, str)):
        if len(drop) != len(X):
          raise Exception('Arrays `drop` and `X` must be of the same length.')
      else:
        if isinstance(drop, float) and np.isnan(drop):
          drop = np.isnan(X)
        else:
          drop = X == drop
    start =0
    max_start = len(X) - window_size - target_size + 1
    while start < max_start:
      end = start + window_size
      if drop_windows:
        drop_window = drop[start:end+target_size]
        to_drop = np.where(drop_window)[0]
        if to_drop.size:
          start += to_drop[-1] + 1
          continue
      out_X.append(X[start:end])
      out_y.append(target[end:end+target_size])
      X_index.append(index[start])
      y_index.append(index[end])
      start = start + step_size
    return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)

## 이상길이 찾는 함수 - 이상치 클래스 Anomaly를 활용하여 예측 이상구간의 길이 구하기
def process_anomaly_detection(X, y_hat, critic, X_index):
    
    ## label 데이터 파일 경로 설정 2
    anomaly_file = "C:/Users/user/BusanDigitalAcademy/Battery_Project/Test07_NG_dchg_Label.csv"
    
    known_anomalies = pd.read_csv(anomaly_file)
    final_scores, true_index, true, predictions = anomaly.score_anomalies(X, y_hat, critic, X_index, comb="mult")

    if final_scores is None:
        return None, None, None
    final_scores = np.array(final_scores)
    print(final_scores.shape)
    anomalies = anomaly.find_anomalies(final_scores, true_index)
    print(anomalies)

    anom_labels = known_anomalies['label']
    true0 = anom_labels
    pred_length = len(final_scores)
    avg, sigma = np.mean(final_scores), np.std(final_scores)
    Z_score1 = (final_scores - avg) / sigma

    pred_bin=[0]*pred_length
    for i in range(len(anomalies)):
      print( anomalies[i][0], anomalies[i][1])
      for k in range(anomalies[i][0]-1, anomalies[i][1]):
        pred_bin[k]=1

    pred = np.array(pred_bin)
    true = []
    true = true0[: pred_length]
    gt = np.array(true)

    anomalies = find_anomalies(gt, pred)

    return anomalies, pred_length, X, Z_score1, final_scores

## anomalies(이상치) 찾는 함수 - 이상길이를 활용하여 label데이터를 활용한 실제 이상구간과 예측 이상구간 도출
def find_anomalies(gt, pred):
    
    # 변수 초기화
    anomalies = []
    anomaly_gt = []  # 실제 이상구간
    anomaly_pred = []  # 예측한 이상구간
    length_anom = len(pred)

    # 이상구간 바깥 -> 0으로 표시
    # 이상구간 내부 -> 1으로 표시
    anom_pred_init = 0  
    anom_gt_init = 0 

    
    for k in range(length_anom):
        
        # 실제 이상구간 탐지(anomaly_gt)
        if gt[k] == 1: # 이상구간 시작되었을 때(label : 1 / gt_pred_init : 0)
            if anom_gt_init == 0:
                anom_gt_begin = k
                anom_gt_init = 1
            else:
                anom_gt_end = k
                if k == length_anom - 1:
                    anomaly_gt.append((anom_gt_begin, anom_gt_end))

        if gt[k] == 0 and anom_gt_init == 1: # 이상구간이 끝났을 때(label : 0 / gt_pred_init : 1)
            anom_gt_end = k - 1
            anomaly_gt.append((anom_gt_begin, anom_gt_end))
            anom_gt_init = 0

        # 예측된 이상구간 탐지(anomaly_pred)
        if pred[k] == 1: # 이상구간이 시작되었을 때(label : 1 / anom_pred_init : 0)
            if anom_pred_init == 0: 
                anom_pred_begin = k # 시작된 시점 K 기록
                anom_pred_init = 1
            else:
                anom_pred_end = k
                if k == length_anom - 1:
                    anomaly_pred.append((anom_pred_begin, anom_pred_end)) # 이상구간(시작된 시점 K ~ 예측구간 전까지)

        if pred[k] == 0 and anom_pred_init == 1: # 이상구간이 끝났을 때(label : 0 / anom_pred_init : 1)
            anom_pred_end = k - 1
            anomaly_pred.append((anom_pred_begin, anom_pred_end))
            anom_pred_init = 0

    anomalies = [anomaly_gt, anomaly_pred]
    return anomalies

## Chart로 데이터 전송하는 함수 1) 이상치 값들을 Flask 서버에서 Client(웹페이지)로 보내기 위해 Json 형식의 데이터로 변환하
def prepare_anomaly_data(anomalies, length_anom, X, Z_score1):
    latest_data_size = min(100, len(X))
    time = list(range(length_anom - latest_data_size, length_anom))
    Z_score2 = Z_score1[-latest_data_size:]
    X_signal = [X[kk, 1] for kk in range(-latest_data_size, 0)]
    X_signal_2 = np.array(X_signal)

    datasets = [
        {
            'label': 'PCA1',
            'data': list(2 * X_signal_2[:, 0])
        },
        {
            'label': 'PCA2',
            'data': list(2 * X_signal_2[:, 1])
        },
        {
            'label': 'Z_score',
            'data': list(Z_score2)
        }
    ]

    # 이상 구간 데이터 처리
    anomaly_gt_data = []
    anomaly_pred_data = []

    for t1, t2 in anomalies[0]:  # 실제 이상 구간
        if t1 < length_anom and t2 >= length_anom - latest_data_size:
            anomaly_gt_data.append({'start': t1, 'end': t2})

    for t1, t2 in anomalies[1]:  # 예측된 이상 구간
        if t1 < length_anom and t2 >= length_anom - latest_data_size:
            anomaly_pred_data.append({'start': t1, 'end': t2})

    return {
        'time': time,
        'datasets': datasets,
        'anomaly_gt': anomaly_gt_data,
        'anomaly_pred': anomaly_pred_data
    }

## Chart로 데이터 전송하는 함수 2) 전압 데이터
def prepare_vol_data(vol_df, total_data_count):
    latest_data_size = 100
    vol_df = vol_df.tail(latest_data_size)
    time = list(range(total_data_count - latest_data_size, total_data_count))

    voltage_datasets = []
    for col in vol_df.columns:
        voltage_datasets.append({
            'label': col,  # 열 이름
            'data': vol_df[col].tolist()  # 해당 열의 데이터
        })

    data = {
        'time': time,
        'datasets': voltage_datasets
    }
    return data

## Chart로 데이터 전송하는 함수 3) 온도 데이터
def prepare_tem_data(tem_df, total_data_count):
    latest_data_size = 100
    tem_df = tem_df.tail(latest_data_size)
    time = list(range(total_data_count - latest_data_size, total_data_count))

    temperature_datasets = []
    for col in tem_df.columns:
        temperature_datasets.append({
            'label': col,  # 열 이름
            'data': tem_df[col].tolist()  # 해당 열의 데이터
        })

    data = {
        'time': time,
        'datasets': temperature_datasets
    }
    return data

