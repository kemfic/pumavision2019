import cv2
import numpy as np

if __name__ == '__main__':
  cap = cv2.VideoCapture(1)
  ret, frame = cap.get()
  cv2.imshow('out', frame)
