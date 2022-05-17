import cv2
import numpy as np
import tensorflow as tf
from movinet import get_model
from collections import deque
import os
import time
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

video_path = "550.mp4"
model_path = 'checkpoint.h5'
model = get_model()
model.load_weights(model_path)
# model.summary()

cap = cv2.VideoCapture(video_path)

inference_queue = deque()
FIRST_INFERENCE = False

while True:
    pred = 0
    ret, frame = cap.read()
    if ret == True:
        loop_time = time.time()

        if len(inference_queue) < 5:
            print('appending right')
            start_enqueue = time.time()
            inference_queue.append(cv2.resize(frame, (172, 172)))
            print(f"enqueue time:{time.time()-start_enqueue}")

        if len(inference_queue) == 5:

            t0 = time.time()
            pred = model(np.array(inference_queue)[np.newaxis])[0][0]

            inference_queue.popleft()

            print(f"serve conf:{1-pred} , inference time = {time.time()-t0}")



            FIRST_INFERENCE =True


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
        if FIRST_INFERENCE and frame is not None:
            image = cv2.putText(frame, f"server confidence {1-pred}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('test', frame)
        cv2.waitKey(1)
        if 0xFF == ord('q'):
            break
        # cv2.waitKey(0)
    else:
        break
    print(f"loop time:{time.time()-loop_time}")
cap.release()
cv2.destroyAllWindows()
    #




