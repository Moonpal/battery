import pandas as pd
import numpy as np
from gan_models import *


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
