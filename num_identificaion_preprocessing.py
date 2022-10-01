import numpy as np
import cv2
import matplotlib.pyplot as plt


def open_picture(windowname, img):

  cv2.imshow(windowname, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()



def addweighted(img1, weight_num1, img2, weight_num2, gamma):

  dst = cv2.addWeighted(img1, weight_num1, img2, weight_num2, gamma)
  cv2.imshow('addweighted', dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return dst


def sobel(img):     

  x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
  y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
  
  absX = cv2.convertScaleAbs(x)
  absY = cv2.convertScaleAbs(y)

  dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

  cv2.imshow('sobel', dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return dst




filename = "./DIP/test.png"
origin_img = cv2.imread(filename)
open_picture('origin', origin_img)

plt.imshow(origin_img)
plt.show()


#image segmentation 
ltop = (990, 930)  #lefttop(x,y)
rtbm = (1430, 980) #rightbottom(x,y)
img_cap = origin_img[ltop[1]:rtbm[1], ltop[0]: rtbm[0]]#image segmentation(img[y,x])
open_picture('segment', img_cap)
cv2.imwrite('./DIP/segmentation_img.png', img_cap)

#RGB to Gray level
gray_img = cv2.cvtColor(img_cap, cv2.COLOR_BGR2GRAY)
open_picture('graylevel', gray_img)

#gaussian low pass filter
#gaussian_img = cv2.GaussianBlur(gray_img, (5,5), 10) #decrease irrevalent feature
#open_picture('gausssian', gaussian_img)

#sobel
#sobel_img = sobel(gaussian_img) #high pass filter that can enhance higher intensity part, eg:contour, edge, corner, etc.


#gaussian
#blurred_img = cv2.GaussianBlur(sobel_img, (9, 9),0) 
#ret, binary_img = cv2.threshold(blurred_img , 170, 255, cv2.THRESH_BINARY)


#dilation and erosion1(erosion can delete detail, dilation can make contour more apparent)

#element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
#element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
#fstdilation_img = cv2.dilate(Binary_img, element2, iterations = 1)
#open_picture('dilation1', fstdilation_img)
#erosion_img = cv2.erode(fstdilation_img, element1, iterations = 1)
#open_picture('erosion', erosion_img)
#sndilation_img = cv2.dilate(erosion_img, element2,iterations = 3)
#open_picture('erosion and dilation',sndilation_img)


#dilation and erosion1(erosion can delete detail, dilation can make contour more apparent)

#kernel = np.ones((3,3),np.uint8)
#ret, thresh_img = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #OTSU+thresthod
#opening_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN,kernel, iterations = 2) #more independent
#fstdilation_img = cv2.dilate(opening_img, kernel, iterations = 1)
#open_picture('dilation1', fstdilation_img)
#erosion_img = cv2.erode(fstdilation_img, kernel, iterations = 1)
#open_picture('erosion', erosion_img)
#sndilation_img = cv2.dilate(erosion_img, kernel, iterations = 3)
#open_picture('erosion and dilation',sndilation_img)


#edge detection:
#way1 canny
low_threshold = 0
high_threshold = 255
edged_img = cv2.Canny(gray_img, low_threshold, high_threshold)
open_picture('canny', edged_img)

#way2findcontour
#ret,gray_img = cv2.threshold(gray_img, 127, 255, 0)
#contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#gray_img = cv2.drawContours(gray_img, contours, -1, (0,255,0), 3)
#cv2.drawContours(gray_img, contours, -1, (0, 0, 255), 3)
#open_picture(gray_img)



#way3 watershed
#gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
#gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
#gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
#ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #OTSU+thresthod
#open_picture(thresh)

# noise removal
#kernel = np.ones((3,3), np.uint8)
#opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2) #more independent
#open_picture(opening)


# sure background area
#sure_bg = cv2.dilate(opening, kernel, iterations=3)
#open_picture(sure_bg)

# Finding sure foreground area
#dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5) #make it different  judged on distance
#open_picture(dist_transform)
#ret, sure_fg = cv2.threshold(dist_transform, 0.45*dist_transform.mean(),255,0) 
#open_picture(sure_fg)


# Finding unknown region
#sure_fg = np.uint8(sure_fg)
#unknown = cv2.subtract(sure_bg, sure_fg)


# Marker labelling
#ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
#markers = markers+1
# Now, mark the region of unknown with zero
#markers[unknown==255] = 0

#markers = cv2.watershed(img_cap, markers)

#img_cap[markers == -1] = [255,0,0]
#open_picture(img_cap)
#cv2.imwrite('./DIP/watershed_img.png', img_cap)

