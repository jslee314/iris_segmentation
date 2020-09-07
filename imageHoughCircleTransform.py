import cv2
import numpy as np

img = cv2.imread("D:/2. data/total_iris/houghCircle/1_1000.png", 0)
img = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# (x−xcenter)2+(y−ycenter)2=r2
# image – 8-bit single-channel image. grayscale image.
# method – 검출 방법. 현재는 HOUGH_GRADIENT가 있음.
# dp – dp=1이면 Input Image와 동일한 해상도.
# minDist – 검출한 원의 중심과의 최소거리. 값이 작으면 원이 아닌 것들도 검출이 되고, 너무 크면 원을 놓칠 수 있음.
# param1 – 내부적으로 사용하는 canny edge 검출기에 전달되는 Paramter
# param2 – 이 값이 작을 수록 오류가 높아짐. 크면 검출률이 낮아짐.
# minRadius – 원의 최소 반지름.
# maxRadius – 원의 최대 반지름.
circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT,
                           dp=1, minDist=30, param1=60, param2=70,
                           minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('img', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("원을 찾지 못했습니다.")


import cv2
src = cv2.imread("D:/test1.png")
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
for i in contours:
    hull = cv2.convexHull(i, clockwise=True)
    cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

