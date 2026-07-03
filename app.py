import os
from http import server
import re
from socket import socket
from textwrap import wrap
from turtle import delay
from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room, emit
import base64
from datetime import datetime
import ssl
import csv
import cv2
import time
from unicode import join_jamos
import numpy as np
import io
from PIL import Image
from collections import deque
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

########################
string_list = {0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ',
               12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ',
               23: 'ㅣ', 24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅚ', 29: 'ㅟ', 30: 'ㅢ'}
finger_queue = deque([])
compare_class = None
none_counter = 0
final_word = None

# YOLO 모델 로드 (파일 누락 시 자동 비활성화)
YOLO_net = None
classes = []
output_layers = []

try:
    YOLO_net = cv2.dnn.readNet("./yolov4-obj_best.weights", "./yolov4-obj.cfg")
    with open("./obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = YOLO_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]
    print("[SYSTEM] YOLO 모델 로드 성공")
except cv2.error:
    print("[SYSTEM] YOLO 모델 파일 누락: 수화 인식 기능을 비활성화합니다.")
#######################

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/signUp")
def signUp():
    return render_template('signUp.html')

@app.route("/room")
def room():
    return render_template('room.html')

@app.route("/story")
def story():
    return render_template('ourStory.html')

# SOCKET.IO 로직
@socketio.on('join_room')
def joinRoom(data):
    join_room(data['roomName'])
    emit('welcome', data, broadcast=True, to=data['roomName'], include_self=False)

@socketio.on('offer')
def sendOffer(data, roomName):
    emit('offer', data, broadcast=True, to=roomName, include_self=False)

@socketio.on('answer')
def asnwer(data, roomName):
    emit('answer', data, broadcast=True, to=roomName, include_self=False)

@socketio.on('ice')
def ice(data, roomName):
    emit('ice', data, broadcast=True, to=roomName, include_self=False)

@socketio.on('sendTTS')
def getTTS(ttsData, roomName):
    emit('streamTTS', ttsData, broadcast=True, to=roomName, include_self=True)

@socketio.on('disconnect')
def disconnecting():
    emit('userLeft', request.sid, broadcast=True, include_self=False)

@socketio.on('connect')
def someoneJoin():
    emit('returnMyId', request.sid, broadcast=True, include_self=True, to=request.sid)

@socketio.on('user_message')
def userMessage(data, roomName):
    emit('user_message_from', data, broadcast=True, to=roomName, include_self=False)

@socketio.on('SignUp')
def signUp(data):
    with open('./static/database/user.csv', 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for line in rdr:
            if(line[0] == data['signId']):
                emit('SignUpRes', 'IDExist', to=request.sid, include_self=True)
                return
    with open('./static/database/user.csv', 'a', encoding='utf-8', newline="\n") as f:
        wr = csv.writer(f)
        wr.writerow([data['signId'], data['signPw'], data['signName'], data['signBirth']])
    emit('SignUpRes', 'DONE', to=request.sid, include_self=True)

@socketio.on('SignIn')
def signIn(data):
    with open('./static/database/user.csv', 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for line in rdr:
            if(line[2] == data['signId'] and line[3] == data['signPw']):
                emit('SignInRes', {'states': 'Success', 'userId': data['signId'], 'userName': line[0]}, to=request.sid, include_self=True)
                return
    emit('SignInRes', {'states': 'Fail'}, to=request.sid, include_self=True)

@socketio.on('signImage')
def signImage(data):
    global finger_queue, compare_class, none_counter, final_word
    userImage = data['userImage']
    userImage = userImage.replace("data:image/png;base64,", '')
    img = base64.b64decode(userImage)
    img = Image.open(io.BytesIO(img))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    final_output = None
    if YOLO_net is not None:
        blob = cv2.dnn.blobFromImage(img, 0.00392, (256, 256), (0, 0, 0), True, crop=False)
        YOLO_net.setInput(blob)
        outs = YOLO_net.forward(output_layers)
        
        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y = int(detection[0]*w), int(detection[1]*h)
                    dw, dh = int(detection[2]*w), int(detection[3]*h)
                    boxes.append([int(center_x-dw/2), int(center_y-dh/2), dw, dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                final_output = str(classes[class_ids[i]])

    if final_output is not None:
        final_output = string_list[int(final_output)]
        if none_counter < 2:
            if compare_class != final_output:
                finger_queue.append(final_output)
                compare_class = final_output
                none_counter = 0
        else:
            none_counter += 1
            compare_class = None
    else:
        none_counter += 1
        compare_class = None

    if none_counter >= 2 and finger_queue:
        final_word = list(finger_queue)
        finger_queue = deque([])
        join = join_jamos(''.join(final_word))
        emit('streamSIGN', {'userId': data['userId'], "userText": join}, broadcast=True, to=data['roomName'], include_self=True)
        final_word = None

@socketio.on_error()
def chat_error_handler(e):
    print('An error has occurred: ' + str(e))

if __name__ == "__main__":
    # host='0.0.0.0'을 주어야 외부 컴퓨터가 내 공인 IP를 통해 들어올 수 있습니다.
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
