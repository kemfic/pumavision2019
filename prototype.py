import cv2
import numpy as np
k_scale = 0.5

h_thresh = (50, 90)
s_thresh = (100, 255)
v_thresh = (80, 255)


def preprocess(img):
  #img = cv2.resize(img, (0,0), k_scale, k_scale)

  kernel = np.ones((7,7), np.uint8)

  img = cv2.erode(img, kernel, 10)
  img = cv2.dilate(img, kernel, 10)

  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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

if __name__ == '__main__':
  cap = cv2.VideoCapture(0)
  while True:
    ret, frame = cap.read()

    img = preprocess(frame)
    mask = get_mask(img)

    cv2.imshow("input", frame)


    cv2.imshow("output", cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_HSV2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
