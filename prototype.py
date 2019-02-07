import cv2
import numpy as np
k_scale = 0.5

h_thresh = (70, 90)
s_thresh = (120, 255)
v_thresh = (230, 255)


def preprocess(img):
  #img = cv2.resize(img, (0,0), k_scale, k_scale)

  img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
  return img

def get_mask(img):
  mask = np.ones_like(img[:,:,0])
  h = img[:,:,0]
  s = img[:,:,1]
  v = img[:,:,2]

  mask[h < h_thresh[0]] = 0
  mask[s < s_thresh[0]] = 0
  mask[v < v_thresh[0]] = 0

  mask[h > h_thresh[1]] = 0
  mask[s > s_thresh[1]] = 0
  mask[v > v_thresh[1]] = 0

  return mask

def get_contours(mask):
  ret, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  return contours, hierarchy

if __name__ == '__main__':
  cap = cv2.VideoCapture(0)
  while True:
    ret, frame = cap.read()

    img = preprocess(frame)
    mask = get_mask(img)

    contours, hierarchy = get_contours(mask)


    cv2.imshow("input", frame)
    cv2.imshow("mask", mask*255)

    out = cv2.drawContours(frame, contours, -1, (50, 0, 255), 3)
    cv2.imshow("out", out)
    cv2.imshow("output", cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_HLS2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
