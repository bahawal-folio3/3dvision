import argparse
import copy
import time

from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

from movinet import get_model

MODEL_VERSION = 'a0'

def process_video(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.25,
    dim: int = 1280,
) -> None:
    """
        reads an input video frame by frame and performs  
        
        source_weights_path: path to weights of saved model
        source_video_path: path to video to be processed
        target_video_path: path to output video file
        confidence_threshold: conf thres to filter output
        dim: input dimensions
        Returns:
        None, create an output video
        """
    dim = 172
    video_path = source_video_path
    model_path = source_weights_path
    model = get_model(MODEL_VERSION)
    model.load_weights(model_path)

    cap = cv2.VideoCapture(video_path)

    inference_queue = deque()
    preds_queue = deque()
    FIRST_INFERENCE = True
    window = 8
    serve_count = 0
    pred = 3
    vote = 3
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cool_down_period = fps * 2
    cooldowntime = 0
    cooldown = False
    out = cv2.VideoWriter(
        target_video_path,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (1280, 720),
    )
    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        """
        this loop preforms inferece on each frame and pools the output for 
        over window size. We use a sliding window to keep output smooth. 
        we initiate a cooldown once a hit is detected since complete action takes tiem
        and this prevents duplication.   
        """
        if ret == True:
            if FIRST_INFERENCE:
                for _ in range(window):
                    inference_queue.append(cv2.resize(frame, (dim, dim)))
                    FIRST_INFERENCE = False

            if len(inference_queue) < window:
                inference_queue.append(cv2.resize(frame, (dim, dim)) / 225.0)

            if len(inference_queue) == window:
                confidence = (
                    model(np.array(inference_queue)[np.newaxis])[0]
                    > confidence_threshold
                )
                pred = np.argmax(confidence)
                inference_queue.popleft()
            if cooldown:
                # initiate a cooldown time if a serve is detection to avoid double makring of same event
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
                    cooldown = True
                    preds_queue = deque([2] * fps)
                else:
                    preds_queue.popleft()

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            # white color in BGR
            color = (255, 255, 255)
            thickness = 2

            loop_end = time.time() - loop_start
            fps_count = int(1 / loop_end)
            if not FIRST_INFERENCE and frame is not None:
                cv2.putText(
                    frame,
                    f"serve confidence: {round(float(confidence[0]),1)}, serve count "
                    + f"{serve_count} fps:{fps_count}",
                    org,
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("test", frame)
            out.write((frame))
            cv2.waitKey(1)
            if 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="action recognition in tennis video"
    )
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.5,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--size", default=256, help="size of the img", type=int,
    )
    args = parser.parse_args()

    process_video(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
    )
