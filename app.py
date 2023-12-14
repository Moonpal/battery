from flask import Flask, render_template, redirect, url_for
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

## arguments 정의
arguments=collections.namedtuple('Args',
 'signal_file timest_form anomaly_file mode aggregate_interval regate_interval')
args=arguments(signal_file='/content/drive/MyDrive/충방전 데이터파일/data/raw_data/test/Test07_NG_dchg.csv',
      timest_form=0,
      anomaly_file='C:/Users/user/BusanDigitalAcademy/batterydata/data/preprocessed/test/Test03_OK_chg_Label.csv',
      mode='predict',
      aggregate_interval=1,
      regate_interval=1)

# 파일 경로
# "C:/Users/user/BusanDigitalAcademy/batterydata/data/preprocessed/test/Test01_OK_chg_Label.csv"
# "C:\Users\user\BusanDigitalAcademy\batterydata\data\preprocessed\test\Test02_OK_dchg_Label.csv"
# "C:\Users\user\BusanDigitalAcademy\batterydata\data\preprocessed\test\Test03_OK_chg_Label.csv"
# 'C:/Users/user/BusanDigitalAcademy/batterydata/data/preprocessed/test/Test07_NG_dchg_Label.csv'


## 8번 파일
# args=arguments(signal_file='/content/drive/MyDrive/충방전 데이터파일/data/raw_data/test/Test07_NG_dchg.csv',
#       timest_form=0,
#       anomaly_file='C:/Users/user/BusanDigitalAcademy/batterydata/data/preprocessed/test/Test08_NG_chg_Label.csv',
#       mode='predict',
#       aggregate_interval=1,
#       regate_interval=1)

#########################################################################################################################
app = Flask(__name__)
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
# vol_acc_df = pd.DataFrame()
# tem_acc_df = pd.DataFrame()

# 초기 데이터 로드 함수
def load_initial_data():
    global accumulated_df
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM test07_ng_dchg ORDER BY Time ASC LIMIT 700")
        initial_data = cursor.fetchall()
        accumulated_df = pd.DataFrame(initial_data, columns=[column[0] for column in cursor.description])
        accumulated_df = accumulated_df.iloc[:, 23:]
    except pymysql.Error as e:
        print(e)
    finally:
        cursor.close()

# 데이터를 전송받는 함수 구축
def send_data():
    global accumulated_df
    global blue_graph_detected

    try:
        cursor = db.cursor()

        # 처음에는 last_time을 None으로 설정하여 가장 처음에 받은 데이터의 시간으로 초기화
        last_time = None

        while True:
            # 데이터베이스에서 다음 데이터 가져오기
            if last_time is None:
                query = "SELECT * FROM test03_ok_chg ORDER BY Time ASC LIMIT 10"
                # query = "SELECT * FROM test08_ng_chg ORDER BY Time ASC LIMIT 10"
                # query = "SELECT * FROM test07_ng_dchg ORDER BY Time ASC LIMIT 1"
            else:
                # query = f"SELECT * FROM test08_ng_chg WHERE Time > '{last_time}' ORDER BY Time ASC LIMIT 10"
                query = f"SELECT * FROM test03_ok_chg WHERE Time > '{last_time}' ORDER BY Time ASC LIMIT 10"
            
            # 쿼리문 실행
            cursor.execute(query)
            data = cursor.fetchone()
            
            # 코드 살짝 수정
            # cursor.execute(query, (last_time,))
            # data = cursor.fetchall()

            if data:
                # 데이터프레임으로 변환
                df = pd.DataFrame([data])

                # 여기에서 데이터를 슬라이싱하여 온도, 전압 데이터로 추출
                sliced_df = df.iloc[:, 23:]
                
    
                # 원본 데이터프레임에 현재 데이터 누적
                accumulated_df = pd.concat([accumulated_df, sliced_df], ignore_index=True)

                # 10개씩 주기적으로 diff_smooth + PCA 수행
                if len(accumulated_df) >= 10 and len(accumulated_df) % 10 == 0:
                # if len(accumulated_df) >= 100 and len(accumulated_df) % 100 == 0:

                    processed_df = diff_smooth_df(accumulated_df.copy(), lags_n=0, diffs_n=0, smooth_n=0)
                    processed_df = do_pca(processed_df, 3)
                    X, index = time_segments_aggregate(processed_df, interval=args.aggregate_interval, time_column='date')
                    X = SimpleImputer().fit_transform(X)
                    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
                    
                    # 초기화
                    y_hat, critic = None, None
                    # if len(X) >= 10:
                    if len(X) >= 100:
                        try:
                            X, y, X_index, y_index = rolling_window_sequences(X, index, window_size=100, target_size=1, step_size=1, target_column=0)
                            y_hat, critic = predict(X)
                            # print(y_hat, critic) 문제 X
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

                            # 전압,온도 데이터 추출
                            vol_df = accumulated_df.iloc[:,:176]
                            tem_df = accumulated_df.iloc[:,176:,]
                            
                            # 시각화 함수 호출
                            image_path_vol = save_vol_plot_to_file(vol_df)
                            image_path_tem = save_tem_plot_to_file(tem_df)
                            # image_path = save_plot_to_file(anomalies, length_anom, X, Z_score1, final_scores)
                            image_path, status = save_plot_to_file(anomalies, length_anom, X, Z_score1)
                            print(status)

                            # 이미지 파일의 경로를 클라이언트에 전송
                            # socketio.emit('update_plot', {'image_path': image_path}, namespace='/test')
                            if status == 'stop':
                                data_transfer_status = 'paused'
                                blue_graph_detected = True
                                # 파란색 그래프 감지 이벤트를 클라이언트에게 발송
                                socketio.emit('blue_graph_detected', namespace='/test')
                                break
                            elif status == 'continue':
                                socketio.emit('update_plot', {'image_path': image_path}, namespace='/test')

                            # socketio.emit('update_plot_vol', {'image_path': image_path_vol}, namespace='/test')
                            # socketio.emit('update_plot_tem', {'image_path': image_path_tem}, namespace='/test')
                            
                    else:
                        # 예외가 발생하거나 데이터가 충분하지 않은 경우에 대한 처리
                        pass

            last_time = data['Time']
            db.commit()

            # 1초 간격으로 데이터 갱신
            socketio.sleep(0.1)

    except pymysql.Error as e:
        print(e)

    finally:
        cursor.close()

load_initial_data()

@app.route('/')
def index():
    if blue_background_detected:
        return redirect(url_for('new_page'))
    return render_template('index.html')

@socketio.on('blue_graph_detected', namespace='/test')
def handle_blue_graph_detected():
    print("Blue graph detected")  # 이벤트 발생 로그
    socketio.emit('blue_graph_detected', namespace='/test')  # 클라이언트에게 이벤트 전송

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    global data_transfer_status
    data_transfer_status = 'running'
    socketio.start_background_task(target=send_data)
    return '', 204

@app.route('/new_page')
def new_page():
    # 파란색 그래프 감지 시 리디렉션되는 페이지
    return render_template('new_page.html')

@app.route('/resume_data')
def resume_data():
    global data_transfer_status
    data_transfer_status = 'running'
    return redirect(url_for('index'))

@app.route('/cancel_data')
def cancel_data():
    global data_transfer_status
    # 데이터 전송 상태를 초기화하거나 다른 필요한 조치를 취합니다.
    data_transfer_status = 'paused'
    # 필요한 경우 다른 초기화 코드를 추가합니다.

    # 사용자를 기본 페이지로 리디렉션합니다.
    return redirect(url_for('index'))


if __name__ == '__main__':
    socketio.run(app, debug=True)

# 백그라운드 스레드에서 데이터를 실시간으로 전송하는 함수 실행
# @socketio.on('connect', namespace='/test')
# def test_connect():
#     print('Client connected')
#     socketio.start_background_task(target=send_data)

# Flask 라우트 설정
# @app.route('/')
# # def index():
# #     return render_template('index.html')
# def index():
#     if data_transfer_status == 'paused':
#         return redirect(url_for('new_page'))
#     return render_template('index.html')

# @app.route('/new_page')
# def new_page():
#     return render_template('new_page.html')

# if __name__ == '__main__':
#     socketio.run(app, debug=True)
