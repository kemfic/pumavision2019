import cv2
import numpy as np


class CargoTracker(object):
  thresh = np.array( [[0, 255],
                     [0, 255],
                     [0,255]] )
  def __init_(self):
    self.img = np.zeros((500,500))
  def pipeline(self, img):
    self.img = self.resize(img)
    #self.img = self.recolor(img)
  
  def resize(self, img):
    img = cv2.resize(img,None, fx=0.2, fy=0.2)
    return img

if __name__ == "__main__":
  ct = CargoTracker()
  img = cv2.imread('img/0.jpg')
  img = np.rot90(img)
  ct.pipeline(img)
  cv2.imshow('output', ct.img)

  k = cv2.waitKey(0) & 0xFF
  if k == 27:
    cv2.destroyAllWindows()
