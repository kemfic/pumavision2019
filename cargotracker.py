import cv2
import numpy as np


class CargoTracker(object):
  min_thresh = np.array( [10,30,100] )
  max_thresh = np.array( [50, 255, 255] )
  def __init_(self):
    self.img = np.zeros((500,500))
  def pipeline(self, img):
    self.img = cv2.resize(img, (500, 500))
    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
    self.mask = cv2.inRange(self.img, self.min_thresh, self.max_thresh)

if __name__ == "__main__":
  ct = CargoTracker()
  img = cv2.imread('img/0.jpg')
  img = np.rot90(img)
  ct.pipeline(img)
  cv2.imshow('output', cv2.cvtColor(ct.img, cv2.COLOR_HLS2BGR))
  cv2.imshow('mask', ct.mask)

  k = cv2.waitKey(0) & 0xFF
  if k == 27:
    cv2.destroyAllWindows()
