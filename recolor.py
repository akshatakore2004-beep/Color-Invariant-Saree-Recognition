import cv2, sys
img = cv2.imread(sys.argv[1])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv[..., 0] = (hsv[..., 0].astype(int) + 60) % 180
cv2.imwrite("recolored.jpg", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
