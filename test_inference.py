import cv2
import numpy as np
import tensorflow as tf
import copy
from movinet import get_model
from collections import deque, Counter
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
dim = 172
video_path = "tennis.mp4"
model_path = 'frozen_layers-a0-f1-8-trainableTrue.h5'
model = get_model('a0')
model.load_weights(model_path)
# model.summary()

cap = cv2.VideoCapture(video_path)

inference_queue = deque()
preds_queue = deque()
FIRST_INFERENCE = True
window = 8
serve_count = 0
serve_preds = 0
serve_conf = 0
pred = 3
vote = 3
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("FPS:", fps)
cool_down_period = fps*2
cooldowntime = 0
cooldown = False
out = cv2.VideoWriter(f'output-{video_path}.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (1280, 720))
while True:
    loop_start = time.time()

    ret, frame = cap.read()
    frame2 = copy.deepcopy(frame)

    if ret == True:
        if FIRST_INFERENCE:
            for i in range(window):
                inference_queue.append(cv2.resize(frame, (dim, dim)))
                FIRST_INFERENCE = False

        if len(inference_queue) < window:
            inference_queue.append(cv2.resize(frame, (dim, dim))/225.0)

        if len(inference_queue) == window:
            t0 = time.time()
            confidence = model(np.array(inference_queue)[np.newaxis])[0]
            pred = np.argmax(confidence)
            # pred = np.argmax(pred)
            inference_queue.popleft()
            print(f"serve conf:{round(float(confidence[0]),2)} inference time: {round(time.time()-t0,3)} ")

        if cooldown:
            print("in cooldown :", cooldowntime)
            cooldowntime += 1
            if cooldowntime > cool_down_period:
                cooldowntime = 0
                cooldown = False

        if len(preds_queue) < fps and not cooldown:
            preds_queue.append(pred)
            print(cooldown)

        if len(preds_queue) == fps and not cooldown:
            counter = Counter(preds_queue)
            vote = counter.most_common()[0][0]
            if vote == 0:
                serve_count += 1
                print("count increased")
                cooldown = True
                preds_queue = deque([2]*fps)
            else:
                preds_queue.popleft()


        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        # # white color in BGR
        color = (255, 255, 255)
        thickness = 2

        loop_end = time.time() - loop_start
        fps_count = int(1/loop_end)
        if not FIRST_INFERENCE and frame is not None:
            image = cv2.putText(frame2, f"serve confidence: {round(float(confidence[0]),1)}, serve count " +
                                       f"{serve_count} fps:{fps_count}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.resize(frame2, (1280, 720))
        cv2.imshow('test', frame)
        out.write((frame2))
        cv2.waitKey(1)
        if 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()





