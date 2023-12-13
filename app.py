from flask import Flask, render_template
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
      anomaly_file='C:/Users/user/BusanDigitalAcademy/batterydata/data/preprocessed/test/Test07_NG_dchg_Label.csv',
      mode='predict',
      aggregate_interval=1,
      regate_interval=1)

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
        cursor.execute("SELECT * FROM test07_ng_dchg ORDER BY Time ASC LIMIT 4400")
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
    # global vol_acc_df
    # global tem_acc_df

    try:
        cursor = db.cursor()

        # 처음에는 last_time을 None으로 설정하여 가장 처음에 받은 데이터의 시간으로 초기화
        last_time = None

        while True:
            # 데이터베이스에서 다음 데이터 가져오기
            if last_time is None:
                query = "SELECT * FROM test07_ng_dchg ORDER BY Time ASC LIMIT 10"
                # query = "SELECT * FROM test08_ng_chg ORDER BY Time ASC LIMIT 10"
                # query = "SELECT * FROM test07_ng_dchg ORDER BY Time ASC LIMIT 1"
            else:
                # query = f"SELECT * FROM test08_ng_chg WHERE Time > '{last_time}' ORDER BY Time ASC LIMIT 10"
                query = f"SELECT * FROM test07_ng_dchg WHERE Time > '{last_time}' ORDER BY Time ASC LIMIT 10"
            
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
                # vol_df = pd.concat([vol_acc_df, vol_df], ignore_index=True)
                # tem_acc_df = pd.concat([tem_acc_df, tem_df], ignore_index=True)
                

                # 이미지 파일의 경로를 클라이언트에 전송 -> 가장 아래로 변경
                # socketio.emit('update_plot_vol', {'image_path': image_path_vol}, namespace='/test')
                # socketio.emit('update_plot_tem', {'image_path': image_path_tem}, namespace='/test')

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
                    if len(X) >= 10:
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
                        anomalies, length_anom, X, Z_score1 = process_anomaly_detection(X, y_hat, critic, X_index)
                        if anomalies is not None:

                            # 전압,온도 데이터 추출
                            vol_df = accumulated_df.iloc[:,:176]
                            tem_df = accumulated_df.iloc[:,176:,]
                            print(vol_df)
                            print(tem_df)

                            
                            # 시각화 함수 호출
                            image_path_vol = save_vol_plot_to_file(vol_df)
                            image_path_tem = save_tem_plot_to_file(tem_df)
                            image_path = save_plot_to_file(anomalies, length_anom, X, Z_score1)
                            
                            # 이미지 파일의 경로를 클라이언트에 전송
                            socketio.emit('update_plot', {'image_path': image_path}, namespace='/test')
                            socketio.emit('update_plot_vol', {'image_path': image_path_vol}, namespace='/test')
                            socketio.emit('update_plot_tem', {'image_path': image_path_tem}, namespace='/test')
                            
                    else:
                        # 예외가 발생하거나 데이터가 충분하지 않은 경우에 대한 처리
                        pass

            last_time = data['Time']
            db.commit()

            # 1초 간격으로 데이터 갱신
            socketio.sleep(0.01)

    except pymysql.Error as e:
        print(e)

    finally:
        cursor.close()

load_initial_data()

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


