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

## 데이터 전처리 함수 2)
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


## 데이터 전처리 함수 3)
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

## 이상길이 찾는 함수
def process_anomaly_detection(X, y_hat, critic, X_index):
    anomaly_file = "C:/Users/user/BusanDigitalAcademy/batterydata/data/preprocessed/test/Test07_NG_dchg_Label.csv"
    known_anomalies = pd.read_csv(anomaly_file)
    final_scores, true_index, true, predictions = anomaly.score_anomalies(X, y_hat, critic, X_index, comb="mult")

    if final_scores is None:
        return None, None, None
###########################################################################################################
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

## anomalies 찾는 함수
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


## 기본 이상치 탐지 그래프 그리기
def save_plot_to_file(anomalies, length_anom, X, Z_score1):
    register_matplotlib_converters()
    np.random.seed(0)
    
    # 파란색 배경이 있는지 확인하는 플래그############################################
    blue_background_detected = False

    if anomalies is not None and not isinstance(anomalies, list):
        anomalies = [anomalies]

    # 현재 날짜 및 시간을 기반으로 폴더 이름 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join('static', current_time)

    # 폴더가 없으면 생성
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 결과 이미지 파일의 경로 설정
    image_path = os.path.join(folder_name, 'plot.png')

    # 결과를 이미지 파일로 저장
    fig = plt.figure(figsize=(30, 12))
    ax = fig.add_subplot(111)
    
    # windsize 10
    # max_len = length_anom - 10
    max_len = length_anom - 100
    time = range(max_len)
    Z_score2 = Z_score1[:max_len]
    X_signal = []

    for kk in range(max_len):
        X_signal.append(X[kk, 1])

    X_signal_2 = np.array(X_signal)
    
    plt.plot(time, 3 * X_signal_2[:, 0], label='3*PCA1')
    plt.plot(time, 3 * X_signal_2[:, 1], label='3*PCA2')
    plt.plot(time, Z_score2, label='Z score')
    plt.legend(loc=0, fontsize=30)
    print("length_anom, max_len:", length_anom, max_len)
    
    colors = ['red'] + ['blue'] * (len(anomalies) - 1)
    
    for i, anomaly in enumerate(anomalies):
        if anomaly is not None and not isinstance(anomaly, list):
            anomaly = list(anomaly[['start', 'end']].itertuples(index=False))
        
        for _, anom in enumerate(anomaly):
            t1 = anom[0]
            t2 = anom[1]
            plt.axvspan(t1, t2, color=colors[i], alpha=0.2)

            # 파란색 배경 감지 확인#########################################
            if colors[i] == 'blue':
                blue_background_detected = True
    

    plt.title(' Test03_OK_chg : Red = True Anomaly, Blue = Predicted Anomaly', size=34)
    plt.ylabel('PCA1, PCA2, Z_score', size=30)
    plt.xlabel('Time', size=30)
    plt.xticks(size=26)
    plt.yticks(size=26)
    plt.xlim([time[0], time[-1]])

    # 결과를 이미지 파일로 저장
    fig.savefig(image_path)
    plt.close(fig)

    # 파란색 배경이 감지되면 'stop' 상태와 함께 이미지 경로 반환
    if blue_background_detected:
        return image_path, 'stop'
    else:
        return image_path, 'continue'
    # return image_path


## 2) fixed_anomalies를 활용한 시각화 그리기
# def save_plot_to_file(anomalies, length_anom, X, Z_score1, final_scores):
#     anomaly = Anomaly()
#     # fixed_threshold 계산중
#     fixed_threshold = anomaly._fixed_threshold(final_scores)
    
#     register_matplotlib_converters()
#     np.random.seed(0)

#     if anomalies is not None and not isinstance(anomalies, list):
#         anomalies = [anomalies]

#     # 현재 날짜 및 시간을 기반으로 폴더 이름 생성
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     folder_name = os.path.join('static', current_time)

#     # 폴더가 없으면 생성
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)

#     # 결과 이미지 파일의 경로 설정
#     image_path = os.path.join(folder_name, 'plot.png')

#     # 결과를 이미지 파일로 저장
#     fig = plt.figure(figsize=(30, 12))
#     ax = fig.add_subplot(111)
    
#     # windsize 10
#     # max_len = length_anom - 10
#     max_len = length_anom - 100
#     time = range(max_len)
#     Z_score2 = Z_score1[:max_len]
#     X_signal = []

#     for kk in range(max_len):
#         X_signal.append(X[kk, 1])

#     X_signal_2 = np.array(X_signal)
    
#     # 임계값을 그래프에 표시
#     plt.axhline(y=fixed_threshold, color='green', linestyle='--', label='Threshold')
#     plt.plot(time, 3 * X_signal_2[:, 0], label='3*PCA1')
#     plt.plot(time, 3 * X_signal_2[:, 1], label='3*PCA2')
#     plt.plot(time, Z_score2, label='Z score')
#     plt.legend(loc=0, fontsize=30)
#     print("length_anom, max_len:", length_anom, max_len)
    
#     colors = ['red'] + ['blue'] * (len(anomalies) - 1)
    
#     for i, anomaly in enumerate(anomalies):
#         if anomaly is not None and not isinstance(anomaly, list):
#             anomaly = list(anomaly[['start', 'end']].itertuples(index=False))
        
#         for _, anom in enumerate(anomaly):
#             t1 = anom[0]
#             t2 = anom[1]
#             plt.axvspan(t1, t2, color=colors[i], alpha=0.2)
    

#     plt.title(' Test07_NG : Red = True Anomaly, Blue = Predicted Anomaly', size=34)
#     plt.ylabel('PCA1, PCA2, Z_score', size=30)
#     plt.xlabel('Time', size=30)
#     plt.xticks(size=26)
#     plt.yticks(size=26)
#     plt.xlim([time[0], time[-1]])

#     # 결과를 이미지 파일로 저장
#     fig.savefig(image_path)
#     plt.close(fig)

#     return image_path

# 오차함수 계싼
# def calculate_errors_at_t(t, z_scores):
#     # t 시간에서의 Z 점수 반환
#     return z_scores[t]

## find_threshold를 활용한 시각화 그리기
# def save_plot_to_file(anomalies, length_anom, X, Z_score1, final_scores):
    
    
#     register_matplotlib_converters()
#     np.random.seed(0)

#     if anomalies is not None and not isinstance(anomalies, list):
#         anomalies = [anomalies]

#     # 현재 날짜 및 시간을 기반으로 폴더 이름 생성
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     folder_name = os.path.join('static', current_time)

#     # 폴더가 없으면 생성
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)

#     # 결과 이미지 파일의 경로 설정
#     image_path = os.path.join(folder_name, 'plot.png')

#     # 결과를 이미지 파일로 저장
#     fig = plt.figure(figsize=(30, 12))
#     ax = fig.add_subplot(111)
    
#     # windsize 10
#     # max_len = length_anom - 10
#     max_len = length_anom - 100
#     time = range(max_len)
#     Z_score2 = Z_score1[:max_len]
#     X_signal = []

#     for kk in range(max_len):
#         X_signal.append(X[kk, 1])

#     X_signal_2 = np.array(X_signal)
    
#     anomaly = Anomaly()

#     threshold = []
#     for t in range(max_len):
#         # fixed_threshold 계산중 -> z_range()에 
#         # 각 시간 t에 대한 오차 데이터 계산
#         # errors_at_t = calculate_errors_at_t(t, Z_score2)  # 이 부분은 실제 오차 계산 방식에 따라 수정 필요
#         thresholds = anomaly._find_threshold(Z_score2, z_range = (0,10))
#         threshold.append(thresholds)

#     # 임계값을 그래프에 표시
#     plt.plot(time, threshold, color='green', linestyle='--', label='Threshold')
#     plt.plot(time, 3 * X_signal_2[:, 0], label='3*PCA1')
#     plt.plot(time, 3 * X_signal_2[:, 1], label='3*PCA2')
#     plt.plot(time, Z_score2, label='Z score')
#     plt.legend(loc=0, fontsize=30)
#     print("length_anom, max_len:", length_anom, max_len)
    
#     colors = ['red'] + ['blue'] * (len(anomalies) - 1)
    
#     for i, anomaly in enumerate(anomalies):
#         if anomaly is not None and not isinstance(anomaly, list):
#             anomaly = list(anomaly[['start', 'end']].itertuples(index=False))
        
#         for _, anom in enumerate(anomaly):
#             t1 = anom[0]
#             t2 = anom[1]
#             plt.axvspan(t1, t2, color=colors[i], alpha=0.2)
    

#     plt.title(' Test07_NG : Red = True Anomaly, Blue = Predicted Anomaly', size=34)
#     plt.ylabel('PCA1, PCA2, Z_score', size=30)
#     plt.xlabel('Time', size=30)
#     plt.xticks(size=26)
#     plt.yticks(size=26)
#     plt.xlim([time[0], time[-1]])

#     # 결과를 이미지 파일로 저장
#     fig.savefig(image_path)
#     plt.close(fig)

#     return image_path

## 전압
def save_vol_plot_to_file(vol_df):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join('static', 'static_vol', current_time)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    image_path = os.path.join(folder_name, 'plot.png')

    fig, ax = plt.subplots(figsize=(30, 12))
    for column in vol_df.columns:
        ax.plot(vol_df.index, vol_df[column], label=column)
    ax.set_title('Voltage Data Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage')
    # ax.legend(loc='upper left')
    fig.savefig(image_path)
    plt.close(fig)

    return image_path


## 온도
def save_tem_plot_to_file(tem_df):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join('static', 'static_tem', current_time)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    image_path = os.path.join(folder_name, 'plot.png')

    fig, ax = plt.subplots(figsize=(30, 12))
    for column in tem_df.columns:
        ax.plot(tem_df.index, tem_df[column], label=column)
    ax.set_title('Temperature Data Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    # ax.legend(loc='upper left')
    fig.savefig(image_path)
    plt.close(fig)

    return image_path
