import cv2
import numpy as np


class TapeTracker(object):
  min_thresh = np.array( [80,0,0] )
  max_thresh = np.array( [90, 255, 255] )
  def __init_(self):
    self.img = np.zeros((500,500))
  def pipeline(self, img):
    self.img = cv2.resize(img, (300,300), cv2.INTER_NEAREST)
    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
    
    self.mask = cv2.inRange(self.img, self.min_thresh, self.max_thresh)
    
    kernel = np.ones((5,5), np.uint8)
    #self.mask = cv2.dilate(self.mask,kernel, iterations=2) 
    
    
    self.cnt, self.hier = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    self.ret = np.copy(self.img)
    self.cnt_f = []
    self.cnt = sorted(self.cnt, key=cv2.contourArea, reverse=True)[:2] # get largest contour
    
    for cnt in self.cnt:
      x,y,w,h = cv2.boundingRect(cnt)
      if w < 0.6*h and cv2.contourArea(cnt) > 10:
        cv2.rectangle(self.ret, (x,y), (x+w, y+h), (0,255,0), 2)
        self.cnt_f.append(cnt)

    M_1 = cv2.moments(self.cnt_f[0])
    cx_1 = int(M_1['m10']/M_1['m00'])
    cy_1 = int(M_1['m01']/M_1['m00'])
    
    M_2 = cv2.moments(self.cnt_f[1])
    cx_2 = int(M_2['m10']/M_2['m00'])
    cy_2 = int(M_2['m01']/M_2['m00'])

    midpoint = ((cx_1+cx_2)//2, (cy_1+cy_2)//2)
    self.error = midpoint[0] - self.img.shape[0]
    print(self.error)
    #cy = int(M['m01']/M['m00'])
    #print(cx - self.img.shape[0]//2)
    #print(cx)
    self.ret = cv2.drawContours(self.ret, self.cnt_f, -1, (150, 150, 255), 2)
    self.ret = cv2.circle(self.ret, (cx_1, cy_1), 2, (150, 155, 255))
    self.ret = cv2.circle(self.ret, (cx_2, cy_2), 2, (150, 155, 255))
    self.ret = cv2.circle(self.ret, midpoint, 2, (150, 255, 255))

if __name__ == "__main__":
  ct = TapeTracker()
  img = cv2.imread('img/1.jpg')
  ct.pipeline(img)
  cv2.imshow('output', cv2.resize(cv2.cvtColor(ct.img, cv2.COLOR_HLS2BGR), (500, 500), cv2.INTER_NEAREST))
  cv2.imshow('mask', cv2.resize(ct.mask, (500,500), cv2.INTER_NEAREST))

  cv2.imshow('contour', cv2.resize(cv2.cvtColor(ct.ret, cv2.COLOR_HLS2BGR), (500, 500), cv2.INTER_NEAREST))
  k = cv2.waitKey(0) & 0xFF
  if k == 27:
    cv2.destroyAllWindows()
