import numpy as np
import cv2
import matplotlib.pyplot as plt

def open_picture(name, img):

  cv2.imshow(name, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  

filename = "C:/Users/Max/Desktop/python/programming/DIP/hw3_coins2.jpg"
img = cv2.imread(filename)
open_picture('origin', img)


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
gray = cv2.blur(gray, (3,3))

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #OTSU+thresthod
open_picture('thresh', thresh)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/thresh.jpg', thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2) #more independent
open_picture('opening', opening)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/opening.jpg', opening)
opening = cv2.GaussianBlur(opening, (3, 3), 0)


# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
open_picture('sure_bg', sure_bg)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/sure_bg.jpg', sure_bg)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5) #make it different  judged on distance
open_picture('dist_transform', dist_transform)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/dist_transform.jpg', dist_transform)
ret, sure_fg = cv2.threshold(dist_transform,0.45*dist_transform.max(),255,0) 
open_picture('sure_fg', sure_fg)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/sure_fg.jpg', sure_fg)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)

img[markers == -1] = [255,0,0]
open_picture('output', img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/output.jpg', img)
