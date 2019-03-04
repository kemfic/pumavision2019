import cv2
import numpy as np
from prototype import *

frame = cv2.imread("img/0.jpg")
pipeline(frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
