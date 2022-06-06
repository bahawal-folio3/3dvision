import cv2 as cv
import numpy as np
y_h1 = 481
y_h2 = 610
x_w1 = 51
x_w2 = 211

img = cv.imread(cv.samples.findFile('image1.png'))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(5,5),0)
edges = cv.Canny(blur,100,150,apertureSize = 3)
cdstP = np.copy(edges)
# cv.imshow('test',edges)
# cv.waitKey(0)
linesP = cv.HoughLinesP(edges[y_h1:y_h2,x_w1:x_w2], 1, np.pi / 180, 50, None, 50, 10)

max_y = (0,0)
min_y = (0,720)
max_x = (0,0)
min_x = (1280,0)
print(len(linesP))
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        x1,y1,x2,y2 = l


        # if x1 < min_x[0]:
        #     min_x = (x1,y1)

        # if x2 > max_x[0]:
        #     max_x = (x2,y2)
        #     min_x = (x1,y1)
        # if y1 < min_y[1]:
        #     min_y = (x1,y1)
        # if y2 > max_y[1]:
        #     max_y = (x2,y2)
    # cv.line(img[y_h1:y_h2,x_w1:x_w2],(x1,y1),(x2,y2),(0,0,255),2)

        # cv.line(img[y_h1:y_h2,x_w1:x_w2], (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)
        # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", img)
        # cv.waitKey(0)

    cv.line(img[y_h1:y_h2,x_w1:x_w2],min_x,max_x,(0,0,255),2)

    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", img)
    cv.waitKey(0)



# max_y = (0,0)
# min_y = (0,720)
# max_x = (0,0)
# min_x = (1280,0)

# for line in lines:
#     rho,theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
    
# 

# # cv.line(img[y_h1:y_h2,x_w1:x_w2],min_x,,(0,0,255),2)
# # cv.line(img[y_h1:y_h2,x_w1:x_w2],(x1,y1),(x2,y2),(0,0,255),2)
# cv.imshow('test',img)
# cv.waitKey(0)
# cv.imwrite('houghlines3.jpg',img)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y