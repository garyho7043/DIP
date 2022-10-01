import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys 


def open_picture(windowname, img):

  cv2.imshow(windowname, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  
  
def img_segmentation(img, ltop, rtbm):
  
  img_cap = img[ltop[1]:rtbm[1], ltop[0]: rtbm[0]]#image segmentation(img[y,x])
  
  return img_cap
  
  
  
  


filename = "C:/Users/Max/Desktop/python/programming/DIP/test.png"
origin_img = cv2.imread(filename)
open_picture('origin', origin_img)


plt.imshow(origin_img)
plt.show()


#image segmentation 
ltop = (997, 930)  #lefttop(x,y)
rtbm = (1430, 980) #rightbottom(x,y)
img_cap = img_segmentation(origin_img, ltop, rtbm)
open_picture('segment', img_cap)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/segmentation_img.png', img_cap)


#RGB to Gray level
gray_img = cv2.cvtColor(img_cap, cv2.COLOR_BGR2GRAY)
open_picture('graylevel', gray_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/gray_img.png', gray_img)




#edge detection:
#canny
low_threshold = 0
high_threshold = 255
edged_img = cv2.Canny(gray_img, low_threshold, high_threshold)
open_picture('canny', edged_img)



plt.imshow(edged_img)
plt.show()


#num segmentation
num1_ltop = (0, 10)  #lefttop(x,y)
num1_rtbm = (21, 48) #rightbottom(x,y)
num1 = img_segmentation(edged_img, num1_ltop, num1_rtbm)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/gray_img0.png', num1)
open_picture('num1', num1)


i = 0
x = 22


while i < 17 : 
  
  numn_ltop = (x, 10)  #lefttop(x,y)
  numn_rtbm = (x+24, 48) #rightbottom(x,y)
  numn = img_segmentation(edged_img, numn_ltop, numn_rtbm)
  x = x + 24
  i= i + 1
  open_picture('num%d'%(i), numn)
  cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/gray_img%d.png'%(i), numn)
  
  
