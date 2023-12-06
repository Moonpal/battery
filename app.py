from flask import Flask, render_template
from flask_socketio import SocketIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from data_processing import *
from gan_models import *
from anomaly_detection import Anomaly

import pandas as pd
import pymysql
import json

app = Flask(__name__)
socketio = SocketIO(app)
anomaly = Anomaly()

# MySQL 연결 설정
db = pymysql.connect(
    host='localhost',
    user='root',
    password='qwer1234',
    database='battery_voltage_temperature',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# 새로운 전역 변수로 누적된 DataFrame 설정
accumulated_df = pd.DataFrame()

# 데이터를 전송받는 함수 구축
def send_data():
    global accumulated_df

    try:
        cursor = db.cursor()

        # 처음에는 last_time을 None으로 설정하여 가장 처음에 받은 데이터의 시간으로 초기화
        last_time = None

        while True:
            # 데이터베이스에서 다음 데이터 가져오기
            if last_time is None:
                query = "SELECT * FROM test07_ng_dchg ORDER BY Time ASC LIMIT 1"
            else:
                query = f"SELECT * FROM test07_ng_dchg WHERE Time > '{last_time}' ORDER BY Time ASC LIMIT 1"
            
            # 쿼리문 실행
            cursor.execute(query)
            # 가져온 데이터를 JSON 형식으로 변환
            data = cursor.fetchone()
     
            if data:
                # 데이터프레임으로 변환
                df = pd.DataFrame([data])

                # 여기에서 데이터를 슬라이싱하여 원하는 열(24번째 열부터)까지 추출
                sliced_df = df.iloc[:, 23:]

                # 원본 데이터프레임에 현재 데이터 누적
                accumulated_df = pd.concat([accumulated_df, sliced_df], ignore_index=True)
                # print(accumulated_df)

                # 10개씩 주기적으로 diff_smooth + PCA 수행
                if len(accumulated_df) >= 10 and len(accumulated_df) % 10 == 0:
                    processed_df = diff_smooth_df(accumulated_df.copy(), lags_n=0, diffs_n=0, smooth_n=0)
                    processed_df = do_pca(processed_df, 3)
                    X, index = time_segments_aggregate(processed_df, interval=1,time_column='date')
                    X = SimpleImputer().fit_transform(X)
                    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
                    # X, y, X_index, y_index=rolling_window_sequences(X, index, window_size=10, target_size=1, step_size=1, target_column=0) # rolling_window_sequence를 적용하기 위해서는 10개가 만족해야하는 조건이 필요!!
                    # 초기화
                    y_hat, critic = None, None
                    if len(X) >= 10:
                        try:
                            X, y, X_index, y_index = rolling_window_sequences(X, index, window_size=10, target_size=1, step_size=1, target_column=0)
                            y_hat, critic = predict(X)
                        except Exception as e:
                            print(f"An error occurred during prediction: {e}")
                            # 예외가 발생한 경우에도 초기화
                            y_hat, critic = None, None
                    
                    # print(X)
                    # print(X.shape) # (10,10,3) 인데 predict(X) 의 X는 몇 shape가 들어가야함? -> (4584, 10, 3 ) 가능
                    # predict 함수 내에서 예측 전에 로그 추가
                    # print("Input to predict function:")
                    # print(X)
                    # print("Shape of input to predict function:")
                    # print(X.shape)


                    # 예외가 발생하지 않은 경우에만 final_scores 계산
                    if y_hat is not None and critic is not None:
                        final_scores, true_index, true, predictions = anomaly.score_anomalies(X, y_hat, critic, X_index, comb="mult")

                        if final_scores is not None:
                            avg = np.average(final_scores)
                            sigma = np.std(final_scores)
                            Z_score1 = (final_scores - avg) / sigma

                            # anomalies(이상치) 찾기
                            pred_length = len(final_scores)
                            pred_bin = [0] * pred_length
                            pred = np.array(pred_bin)

                            # length_anom 찾기
                            length_anom = len(pred)

                            gt = np.array(true)
                            anomalies = find_anomalies(gt, pred)
                            print(anomalies)

                            # 시각화 함수 호출
                            visualize_anomalies(anomalies, length_anom, X, Z_score1)
                    else:
                        # 예외가 발생하거나 데이터가 충분하지 않은 경우에 대한 처리
                        final_scores, true_index, true, predictions = None, None, None, None
                    # 예외가 발생하지 않은 경우에만 final_scores 계산
                    # if y_hat is not None and critic is not None:
                    #     final_scores, true_index, true, predictions = anomaly.score_anomalies(X, y_hat, critic, X_index, comb="mult")
                    # else:
                    #     # 예외가 발생하거나 데이터가 충분하지 않은 경우에 대한 처리
                    #     final_scores, true_index, true, predictions = None, None, None, None
                                        
                    # # final_scores, true_index, true, predictions = anomaly.score_anomalies(X, y_hat, critic, X_index, comb="mult")
                    # # processed_df = pd.DataFrame(X)
                    # # print(processed_df)
                    # # z_score1 찾기
                    # avg = np.average(final_scores)
                    # sigma = math.sqrt(sum((final_scores-avg) * (final_scores-avg)) /len(final_scores))
                    # Z_score1 = (final_scores-avg) / sigma

                    # # anomalies(이상치) 찾기
                    # pred_length =len(final_scores)
                    # pred_bin=[0]*pred_length
                    # pred = np.array(pred_bin)
                    
                    # # length_anom 찾기
                    # length_anom =len(pred)

                    # gt = np.array(true)
                    # anomalies = find_anomalies(gt, pred)
                    
                    # # 시각화 함수 호출
                    # visualize_anomalies(anomalies, length_anom, X, Z_score1)



                    # 전처리된 데이터를 JSON 형식으로 변환(고민해보기!)
                    json_data = processed_df.to_json(orient='records')

                    # 소켓을 통해 데이터를 모든 클라이언트로 전송
                    socketio.emit('update_table', {'data': json_data}, namespace='/test', room=None, skip_sid=None)

            last_time = data['Time']
            db.commit()

            # 1초간격으로 데이터 갱신
            socketio.sleep(1)

    except pymysql.Error as e:
        print(e)

    finally:
        cursor.close()

# 백그라운드 스레드에서 데이터를 실시간으로 전송하는 함수 실행
@socketio.on('connect', namespace='/test')
def test_connect():
    print('Client connected')
    socketio.start_background_task(target=send_data)

# Flask 라우트 설정
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)


