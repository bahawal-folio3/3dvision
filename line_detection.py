import cv2 
import numpy as np
from time import time

y_max = 15
def find_point(lines):
    global y_max
    print(y_max)
    point = []
    for line in lines[0]:
        if line[1] > y_max:
            y_max = line[1]
            point = [line[0],line[1]]
    return point

def main():
    video_path = '550.mp4'
    cap = cv2.VideoCapture(video_path)
    line = [[0,0],[0,0]]
    count = 0

    y_h1 = 505
    y_h2 = 585
    x_w1 = 94
    x_w2 = 191


    # y_h1 = 480
    # y_h2 = 520
    # x_w1 = 190
    # x_w2 = 1050

    while cap.isOpened():
        t0 = time()

        ret, img = cap.read()
        if ret:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            edges = cv2.Canny(blur,100,150,apertureSize = 3)
            cdstP = np.copy(edges)
            linesP = cv2.HoughLinesP(edges[y_h1:y_h2,x_w1:x_w2], 1, np.pi / 180, 50, None, 50, 10)
            # cv2.imshow("edges", img[y_h1:y_h2, x_w1:x_w2])
            if linesP is not None:
                # for i in range(0, len(linesP)):
                l = find_point(linesP)
                # print(l)
                # break
                try:
                    cv2.line(img, (x_w2, y_h1), (x_w1+l[0], l[1]+y_h1), (0, 0, 255), 2, cv2.LINE_AA)
                except IndexError as e:
                    raise(e)


                cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()