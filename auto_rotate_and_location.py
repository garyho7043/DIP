import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys 
import imutils
from scipy.stats import gaussian_kde

def open_picture(windowname, img):

  cv2.imshow(windowname, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  
  
def img_segmentation(img, ltop, rtbm):
  
  img_cap = img[ltop[1]:rtbm[1], ltop[0]: rtbm[0]]#image segmentation(img[y,x])
  
  return img_cap
  
  
  
  

#open
filename = "C:/Users/Max/Desktop/python/programming/DIP/107422.jpg"
origin_img = cv2.imread(filename)
#open_picture('origin', origin_img)

#resize
small_img = imutils.resize(origin_img, width=640)
#open_picture('small', small_img)


#add a white frame
padding = int(origin_img.shape[1]/25)
padding_img = cv2.copyMakeBorder(origin_img, padding, padding, padding, padding,
      cv2.BORDER_CONSTANT, value=[255, 255, 255])
#open_picture('padding', padding_img)




# turn to HSV
hsv_img = cv2.cvtColor(padding_img, cv2.COLOR_BGR2HSV)
hsv_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)





saturation_img = hsv_img[:,:,1]
saturation_small_img = hsv_small_img[:,:,1]

value_img = hsv_img[:,:,2]
value_small_img = hsv_small_img[:,:,2]

sv_ratio = 0.5#
sv_value_img = cv2.addWeighted(saturation_img, sv_ratio, value_img, 1-sv_ratio, 0)
sv_value_small_img = cv2.addWeighted(saturation_small_img, sv_ratio, value_small_img, 1-sv_ratio, 0)




# pit to adjust ratio
plt.subplot(131).set_title("Saturation"), plt.imshow(saturation_img), plt.colorbar()
plt.subplot(132).set_title("Value"), plt.imshow(value_img), plt.colorbar()
plt.subplot(133).set_title("SV-value"), plt.imshow(sv_value_img), plt.colorbar()
plt.show()



#open_picture('padding', sv_value_img)
#open_picture('padding', sv_value_small_img)


#find threshold value
density = gaussian_kde(sv_value_small_img.ravel(), bw_method=0.25)#

step = 0.5
xs = np.arange(0, 256, step)
ys = density(xs)
cum = 0
for i in range(1, 250):
  cum += ys[i-1] * step
  if (cum > 0.02) and (ys[i] < ys[i+1]) and (ys[i] < ys[i-1]):
    threshold_value = xs[i]
    break

print(threshold_value)



plt.hist(sv_value_small_img.ravel(), 256, [0, 256], True, alpha=0.5)
plt.plot(xs, ys, linewidth = 2)
plt.axvline(x=threshold_value, color='r', linestyle='--', linewidth = 2)
plt.xlim([0, max(threshold_value*2, 80)])
plt.show()






#threholding

_, threshold = cv2.threshold(sv_value_img, threshold_value, 255.0, cv2.THRESH_BINARY)

#denoise
kernel_radius = int(padding_img.shape[1]/100)
kernel = np.ones((kernel_radius, kernel_radius), np.uint8)
threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)


plt.imshow(threshold, "gray")
plt.show()











# findcontour
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


debug_img = padding_img.copy()


line_width = int(padding_img.shape[1]/100)


cv2.drawContours(debug_img, contours, -1, (255, 0, 0), line_width)


c = max(contours, key = cv2.contourArea)


x, y, w, h = cv2.boundingRect(c)
cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), line_width)


rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(debug_img, [box], 0, (0, 0, 255), line_width)

print(rect)


plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
plt.show()


#rotate
angle = rect[2]
print(angle)
if angle < -45:
  angle = 90 + angle


(h, w) = padding_img.shape[:2]
center = (w // 2, h // 2)


M = cv2.getRotationMatrix2D(center, angle, 1.0)


rotated = cv2.warpAffine(debug_img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
img_final = cv2.warpAffine(padding_img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.show()











#plt.imshow(origin_img)
#plt.show()




#auto_rotate 


























#image segmentation 
#ltop = (997, 930)  #lefttop(x,y)
#rtbm = (1430, 980) #rightbottom(x,y)
#img_cap = img_segmentation(origin_img, ltop, rtbm)
#open_picture('segment', img_cap)
#cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/segmentation_img.png', img_cap)


#RGB to Gray level
#gray_img = cv2.cvtColor(img_cap, cv2.COLOR_BGR2GRAY)
#open_picture('graylevel', gray_img)
#cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/gray_img.png', gray_img)

