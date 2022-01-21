import numpy as np
import cv2


def open_picture(img):

  cv2.imshow('My Image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def laplacian(img):     

  gray_lap = cv2.Laplacian(img, cv2.CV_16S)
  dst = cv2.convertScaleAbs(gray_lap) 
  cv2.imshow('laplacian', dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return dst

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

  cv2.imshow("absX", absX)
  cv2.imshow("absY", absY)
  cv2.imshow("Result", dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return dst


def five_mul_five_box_filter(img):     

  dst = cv2.blur(img, (5, 5))
  cv2.imshow('five_mul_five_box_filter', dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return dst

def product(img1, img2):
  
  dst = cv2.multiply(img1, img2)
  cv2.imshow('product', dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return dst




def gammatransformation(img):
  dst = np.array(255*(img / 255) ** 0.5, dtype = 'uint8')
  cv2.imshow('gamma_transformation', dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return dst








filename = "C:/Users/Max/Desktop/python/programming/DIP/bonescan.tif"
origin_img = cv2.imread(filename)

open_picture(origin_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/3.57(a).jpg', origin_img)


#3.57(b)
laplacian_img = laplacian(origin_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/3.57(b).jpg', laplacian_img)

#3.57(c)
weightnum1 = 0.5
weightnum2 = 0.5
gamma = 0
addweighted_img = addweighted(origin_img, weightnum1, laplacian_img, weightnum2, gamma)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/3.57(c).jpg', addweighted_img)

#3.57(d)
sobel_img = sobel(origin_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/3.57(d).jpg', sobel_img)


#3.57(e)
five_mul_five_box_filter_img = five_mul_five_box_filter(sobel_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/3.57(e).jpg', five_mul_five_box_filter_img)

#3.57(f)
product_img = product(laplacian_img, five_mul_five_box_filter_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/3.57(f).jpg', product_img)

#3.57(g)
weightnum1 = 0.5
weightnum2 = 0.5
gamma = 0
addweighted_img_two = addweighted(origin_img, weightnum1, product_img, weightnum2, gamma)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/3.57(g).jpg', addweighted_img_two)

#3.57(h)
gamma_transformation_img = gammatransformation(addweighted_img_two)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/3.57(h).jpg', gamma_transformation_img)


