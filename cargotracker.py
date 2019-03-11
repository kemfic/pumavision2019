import cv2
import numpy as np


class CargoTracker(object):
  min_thresh = np.array( [5,50,200] )
  max_thresh = np.array( [40, 255, 255] )
  def __init_(self):
    self.img = np.zeros((500,500))
  def pipeline(self, img):
    self.img = cv2.resize(img, (150, 150), cv2.INTER_NEAREST)
    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
    
    self.mask = cv2.inRange(self.img, self.min_thresh, self.max_thresh)
    
    kernel = np.ones((5,5), np.uint8)
    #self.mask = cv2.dilate(self.mask,kernel, iterations=2) 
    
    
    self.cnt, self.hier = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    self.cnt = sorted(self.cnt, key=cv2.contourArea, reverse=True)[0] # get largest contour

    M = cv2.moments(self.cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    print(cx - self.img.shape[0]//2)
    print(cx)
    self.ret = np.copy(self.img)
    self.ret = cv2.drawContours(self.ret, self.cnt, -1, (150, 100, 255), 3)
    self.ret = cv2.circle(self.ret, (cx, cy), 2, (150, 100, 255))

if __name__ == "__main__":
  ct = CargoTracker()
  img = cv2.imread('img/0.jpg')
  img = np.rot90(img)
  ct.pipeline(img)
  cv2.imshow('output', cv2.resize(cv2.cvtColor(ct.img, cv2.COLOR_HLS2BGR), (500, 500), cv2.INTER_NEAREST))
  cv2.imshow('contour', cv2.resize(cv2.cvtColor(ct.ret, cv2.COLOR_HLS2BGR), (500, 500), cv2.INTER_NEAREST))
  cv2.imshow('mask', cv2.resize(ct.mask, (500,500), cv2.INTER_NEAREST))

  k = cv2.waitKey(0) & 0xFF
  if k == 27:
    cv2.destroyAllWindows()
