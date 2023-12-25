# 초기 라이브리러 불러올 코드
from flask import Flask, render_template, redirect, url_for, session, request
from flask_socketio import emit
from flask_socketio import SocketIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from data_processing import *
from gan_models import *
from anomaly_detection import Anomaly

from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

import pandas as pd
import pymysql  
import threading

## arguments 정의
arguments=collections.namedtuple('Args',
 'signal_file timest_form anomaly_file mode aggregate_interval regate_interval')
args=arguments(signal_file='',
      timest_form=0,
      
      ## label 데이터 파일 경로 설정 1
      anomaly_file='C:/Users/user/BusanDigitalAcademy/Battery_Project/Test07_NG_dchg_Label.csv',
      
      mode='predict',
      aggregate_interval=1,
      regate_interval=1)
      
app = Flask(__name__)
app.static_folder = 'static'
socketio = SocketIO(app)

# 전역 변수로 데이터 전송 상태 관리
data_transfer_status = 'paused'  # 초기 상태는 중단

# 전역 변수로 파란색 그래프 감지 상태 관리
blue_background_detected = False

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
total_data_count = 0

# 초기 데이터 로드 함수
# def load_initial_data():
#     global accumulated_df
#     global data_transfer_status
#     global total_data_count
#     total_data_count = 4100 # 아래 LIMIT 4400과 함께 수정
    
#     try:
#         cursor = db.cursor()
#         cursor.execute("SELECT * FROM test07_ng_dchg ORDER BY Time ASC LIMIT 4100")
#         initial_data = cursor.fetchall()
#         accumulated_df = pd.DataFrame(initial_data, columns=[column[0] for column in cursor.description])
#         accumulated_df = accumulated_df.iloc[:, 23:]
#     except pymysql.Error as e:
#         print(e)
#     finally:
#         cursor.close()

# 데이터를 전송받는 함수 구축
def send_data():
    global accumulated_df
    global blue_graph_detected
    global total_data_count

    try:
        cursor = db.cursor()

        # 처음에는 last_time을 None으로 설정하여 가장 처음에 받은 데이터의 시간으로 초기화
        last_time = None

        while True:
            # 데이터베이스에서 다음 데이터 가져오기(10 묶음)
            if last_time is None:
                query = "SELECT * FROM test07_ng_dchg ORDER BY Time ASC LIMIT 10"
            else:
                query = f"SELECT * FROM test07_ng_dchg WHERE Time > '{last_time}' ORDER BY Time ASC LIMIT 10"
            
            # 쿼리문 실행
            cursor.execute(query)
            data = cursor.fetchone()

            if data:
                # 데이터프레임으로 변환
                df = pd.DataFrame([data])

                # 여기에서 데이터를 슬라이싱하여 온도, 전압 모두 포함된 데이터로 추출
                sliced_df = df.iloc[:, 23:]
                
    
                # 원본 데이터프레임에 현재 데이터 누적
                accumulated_df = pd.concat([accumulated_df, sliced_df], ignore_index=True)
                
                # 전압,온도 데이터 추출
                vol_df = accumulated_df.iloc[:,:176]
                tem_df = accumulated_df.iloc[:,176:]

                # total_data_count 업데이트
                total_data_count += len(sliced_df)

                # 10개씩 주기적으로 diff_smooth + PCA 수행
                if len(accumulated_df) >= 10 and len(accumulated_df) % 10 == 0:

                    processed_df = diff_smooth_df(accumulated_df.copy(), lags_n=0, diffs_n=0, smooth_n=0)
                    processed_df = do_pca(processed_df, 3)
                    X, index = time_segments_aggregate(processed_df, interval=args.aggregate_interval, time_column='date')
                    X = SimpleImputer().fit_transform(X)
                    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
                    
                    # 초기화
                    y_hat, critic = None, None
                    if len(X) >= 10:
                        try:
                            X, y, X_index, y_index = rolling_window_sequences(X, index, window_size=100, target_size=1, step_size=1, target_column=0)
                            y_hat, critic = predict(X)
                            anomaly = Anomaly()
                        except Exception as e:
                            print(f"An error occurred during prediction: {e}")
                            # 예외가 발생한 경우에도 초기화
                            y_hat, critic = None, None


                    # 예외가 발생하지 않은 경우에만 final_scores 계산
                    if y_hat is not None and critic is not None:
                        anomalies, length_anom, X, Z_score1, final_scores = process_anomaly_detection(X, y_hat, critic, X_index)
                        
                        # fixed_threshold 계산중
                        # fixed_threshold = anomaly._fixed_threshold(final_scores)
                        if anomalies is not None:
                            print(anomalies)
                            # # 전압,온도 데이터 추출
                            vol_df = accumulated_df.iloc[:,:176]
                            tem_df = accumulated_df.iloc[:,176:]
                            print(vol_df)
                            print(tem_df)
                            
                            columns_with_values_over_4 = []

                            # 각 열에 대해 반복
                            for column in vol_df:
                                if any(vol_df[column] > 4):
                                    columns_with_values_over_4.append(column)

                            print(columns_with_values_over_4)

                            anomaly_data = prepare_anomaly_data(anomalies, length_anom, X, Z_score1)
                            # print(anomaly_data)
                            vol_data = prepare_vol_data(vol_df, total_data_count)
                            tem_data = prepare_tem_data(tem_df, total_data_count)

                            socketio.emit('update_anomaly_data', anomaly_data, namespace='/test')
                            socketio.emit('update_vol_data', vol_data, namespace='/test')
                            socketio.emit('update_tem_data', tem_data, namespace='/test')

                    else:
                        # 예외가 발생하거나 데이터가 충분하지 않은 경우에 대한 처리
                        pass

            last_time = data['Time']
            db.commit()

            # 1초 간격으로 데이터 갱신 (혹은 숫자 조절을 통해 빠른 주기의 데이터 갱신)
            socketio.sleep(1)

    except pymysql.Error as e:
        print(e)

    finally:
        cursor.close()

# 초기 데이터를 불러올 경우 실행코드
# load_initial_data()

# 백그라운드 스레드에서 데이터를 실시간으로 전송하는 함수 실행
# Flask 라우트 설정
@app.route('/')
def index():
    # 기본 홈페이지를 렌더링합니다.
    return render_template('dashboard.html')

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    # MTadGAN 알고리즘 적용 분석을 시작하는 로직
    global data_transfer_status
    data_transfer_status = 'running'
    socketio.start_background_task(target=send_data)
    return '', 204

@socketio.on('connect', namespace='/test')
def test_connect():
    # 클라이언트가 Socket.IO를 통해 연결되었을 때 로그 확인
    print('Client connected via Socket.IO')

if __name__ == '__main__':
    socketio.run(app, debug=True)
