import cv2
import numpy as np
import tensorflow as tf
from movinet import get_model
from collections import deque, Counter
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
dim = 172
video_path = "550.mp4"
model_path = 'a4-f1-8-trainableFalse.h5'
model = get_model('a4')
model.load_weights(model_path)
# model.summary()

cap = cv2.VideoCapture(video_path)

inference_queue = deque()
preds_queue = deque()
FIRST_INFERENCE = True
window = 8
while True:
    pred = 1
    vote = 1


    ret, frame = cap.read()
    if ret == True:
        loop_time = time.time()

        if FIRST_INFERENCE:
            for i in range(window):
                inference_queue.append(cv2.resize(frame, (dim, dim)))

        if len(inference_queue) < window:
            inference_queue.append(cv2.resize(frame, (dim, dim)))

        if len(inference_queue) == window:


            pred = model(np.array(inference_queue)[np.newaxis])[0][1]
            print(pred)

            inference_queue.popleft()


            print(f"serve conf:{pred} ")



            FIRST_INFERENCE = False
        if len(preds_queue) < 8:
            preds_queue.append(int(pred))
        if len(preds_queue) == 8:

            counter = Counter(preds_queue)
            vote = counter.most_common()[0][0]
            preds_queue.popleft()


        # Window name in which image is displayed
        # window_name = 'Image'
        #
        # # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        #
        # # org
        org = (50, 50)
        #
        # # fontScale
        fontScale = 1
        #
        # # Blue color in BGR
        color = (255, 0, 0)
        #
        # # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        if not FIRST_INFERENCE and frame is not None:
            image = cv2.putText(frame, f"server confidence: {vote}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('test', frame)
        cv2.waitKey(1)
        if 0xFF == ord('q'):
            break
        # cv2.waitKey(0)
    else:
        break
    # print(f"loop time:{time.time()-loop_time}")
cap.release()
cv2.destroyAllWindows()





