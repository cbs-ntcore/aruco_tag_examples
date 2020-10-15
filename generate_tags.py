import cv2
import numpy


dd = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

s = 100
w = 5
h = 10

canvas = numpy.ones(((h * 2 + 1) * s, (w * 2 + 1) * s), dtype='uint8') * 255
for y in range(h):
    for x in range(w):
        i = x + y * w
        print("ID: %s" % i)
        m = cv2.aruco.drawMarker(dd, i, s)
        yp = (y * 2 + 1) * s
        xp = (x * 2 + 1) * s
        canvas[yp:yp+s, xp:xp+s] = m

cv2.imwrite('markers_4x4_50.png', canvas)
