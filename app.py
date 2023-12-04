from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Timer
from data_processing import diff_smooth_df, do_pca, time_segments_aggregate, rolling_window_sequences
import pandas as pd
import pymysql
import json

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
                query = "SELECT * FROM test01_ok_chg ORDER BY Time ASC LIMIT 1"
            else:
                query = f"SELECT * FROM test01_ok_chg WHERE Time > '{last_time}' ORDER BY Time ASC LIMIT 1"
            
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
                print(accumulated_df)

                # 10개씩 주기적으로 PCA 수행
                if len(accumulated_df) >= 10 and len(accumulated_df) % 10 == 0:
                    processed_df = diff_smooth_df(accumulated_df.copy(), lags_n=0, diffs_n=0, smooth_n=0)
                    processed_df = do_pca(processed_df, 3)
                    print(processed_df)

                    # 전처리된 데이터를 JSON 형식으로 변환
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

##############################################################################################################################
# # 데이터를 전송받는 함수 구축
# def send_data():
#     global accumulated_df

#     try:
#         cursor = db.cursor()

#         # 처음에는 last_time을 None으로 설정하여 가장 처음에 받은 데이터의 시간으로 초기화
#         last_time = None

#         while True:
#             # 데이터베이스에서 다음 데이터 가져오기
#             if last_time is None:
#                 query = "SELECT * FROM test01_ok_chg ORDER BY Time ASC LIMIT 1"
#             else:
#                 query = f"SELECT * FROM test01_ok_chg WHERE Time > '{last_time}' ORDER BY Time ASC LIMIT 1"
            
#             # 쿼리문 실행
#             cursor.execute(query)
#             # 가져온 데이터를 JSON 형식으로 변환
#             data = cursor.fetchone()
     
#             if data:
#                 # 데이터프레임으로 변환
#                 df = pd.DataFrame([data])

#                 # 여기에서 데이터를 슬라이싱하여 원하는 열(24번째 열부터)까지 추출
#                 sliced_df = df.iloc[:, 23:]

#                 # 원본 데이터프레임에 현재 데이터 누적
#                 accumulated_df = pd.concat([accumulated_df, sliced_df], ignore_index=True)
#                 print(accumulated_df)

#                  # 10초 이후부터 1초 간격으로 PCA 수행
#                 if len(accumulated_df) > 10 and (len(accumulated_df) - 10) % 10 == 0:
#                     processed_df = diff_smooth_df(accumulated_df.copy(), lags_n=0, diffs_n=0, smooth_n=0)
#                     processed_df = do_pca(processed_df, 3)
#                     print(processed_df)

#                     # time_segment 함수 적용

#                     # 전처리된 데이터를 JSON 형식으로 변환
#                     json_data = processed_df.to_json(orient='records')
                    

#                     # 소켓을 통해 데이터를 모든 클라이언트로 전송
#                     socketio.emit('update_table', {'data': json_data}, namespace='/test', room=None, skip_sid=None)

#             last_time = data['Time']
#             db.commit()

#             # 1초간격으로 데이터 갱신
#             socketio.sleep(1)

#     except pymysql.Error as e:
#         print(e)

#     finally:
#         cursor.close()
##############################################################################################################################

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


