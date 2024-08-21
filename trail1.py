import cv2
import numpy as np
img = cv2.imread('F:\Focus and Non-Focus\sample images//nf1.jpg')

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def angle_cos(p0, p1, p2):
    """Calculate the cosine of the angle between vectors p0p1 and p1p2"""
    d1 = p0 - p1
    d2 = p2 - p1
    return abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours , _ = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    #print(cnt)
                    a = (cnt[1][1] - cnt[0][1])

                    if max_cos < 0.1 and a < img.shape[0]*0.8:

                        squares.append(cnt)
    return squares


squares = find_squares(img_gray)
print(squares[1])
# print(img[squares[1][0][1],squares[1][0][0]])
# print(img.shape)
# print(img[squares[1]])
cv2.drawContours(img_gray, squares, 0, (0, 255, 0), 1)

# Display the result
cv2.imshow('Detected Square', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
