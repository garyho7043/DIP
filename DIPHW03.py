import numpy as np
import cv2
from math import sqrt,exp


def open_picture(img):

  cv2.imshow('My Image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()



def pad_image(img):
  
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  (h,w) = img_gray.shape[:2]
  img_gray = cv2.resize(img_gray, (w, h))
  #open_picture(img_gray)
  resize_img = cv2.resize(img_gray, (2*w, 2*h))
  #open_picture(resize_img)
  
  img_gray = np.array(img_gray)
  resize_img = np.array(resize_img)
  
  for i in range(0, 2*h):
    for j in range(0, 2*w):
        resize_img[i][j] = 0
        
  for i in range(0, h):
    for j in range(0, w):
        resize_img[i][j] = img_gray[i][j]
 
  return  resize_img


def centered_for_FF(img):
  
  (h,w) = img.shape[:2]
  img = np.array(img)
  
  for i in range(0, h):
    for j in range(0, w):
        if ((-1)**(i+j))*(img[i][j]) < 0 :
          img[i][j] = 0
        else :
          img[i][j] = ((-1)**(i+j))*(img[i][j])
        
  
  return img



def DFF(img):
  
  #original = np.fft.fft2(img)
  #center = np.fft.fftshift(original)
  #magnitude_spectrum = 20*np.log(1+np.abs(original))
                          
  DFF_img = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
  dft_shift = DFF_img
  magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
 
  
  return  magnitude_spectrum
  

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)






def gaussian_filter(D0, imgShape):
  
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(0,cols):
        for y in range(0,rows):
            base[y,x] = 250*exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    
    
    
    return base
  
  
  
  
  #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #(h,w) = img_gray.shape[:2]
  #resize_img = cv2.resize(img_gray, (4*w, 4*h))
  #resize_img2 = cv2.resize(img_gray, (2*w, 2*h))
  
  #resize_img = np.array(resize_img)
  
  #for i in range(0, 2*h):
    #for j in range(0, 2*w):
       # resize_img[i][j] = 0

  
  #scale = 3
  #sigma = 1
  #kernel = np.zeros((scale, scale))
  #k_len_max = int(scale / 2)
  #two_sigma_sq = 2.0 * np.power(sigma, 2)
  #one_over_two_pi_sigma_sq = 1.0 / (two_sigma_sq * np.pi)

  #for x in range(-k_len_max, k_len_max + 1):
     # x_sq = np.power(x, 2)
     # for y in range(-k_len_max, k_len_max + 1):
      #  y_sq = np.power(y, 2)
      #  power = -1 * (x_sq + y_sq) / two_sigma_sq
      #  kernel[x + k_len_max][y + k_len_max] = np.exp(power) * one_over_two_pi_sigma_sq
  
  #k_sum = np.sum(kernel)
  
  #for i in range((h-1), (h+2)):
    #for j in range((w-1), (w+2)):    
      #  resize_img[i][j] = kernel[i-h][j-w]
  
  
 # for i in range(0, 2*h):
    #for j in range(0, 2*w):
        #resize_img2[i][j] = resize_img[i][j]
 
  
 # resize_img2 = DFF(resize_img2)
        
  #open_picture(resize_img2)  

  #return resize_img2
  
  

def elementwise(img1, img2):

  elementwise_img = img1
  img1 = np.array(img1)
  img2 = np.array(img2)
  (h,w) = img1.shape[:2]
  
  for i in range(0, h):
    for j in range(0, w):
      elementwise_img[i][j] = img1[i][j] * img2[i][j]
      
  return elementwise_img

def IDFT(img):
 
  #LowPass = np.fft.ifftshift(img)
  #IDFT_img = np.fft.ifft2(img)
  IDFT_img = cv2.idft(img, flags = cv2.DFT_REAL_OUTPUT)
  IDFT_img = centered_for_FF(IDFT_img)
  return IDFT_img

def extract(img1, img2):
  
  (h,w) = img1.shape[:2]
  img2 = np.array(img2)
  
  for i in range(0, h):
    for j in range(0, w):
        img1[i][j] = img2[i][j]
        
  extracted_img = img1
  return extracted_img




filename = "C:/Users/Max/Desktop/python/programming/DIP/integrated-ckt-damaged.tif"
origin_img = cv2.imread(filename)

open_picture(origin_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/4.35(a).jpg', origin_img)

#4.35(b)
zero_ped_img = pad_image(origin_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/4.35(b).jpg', zero_ped_img)

#4.35(c)
centered_for_FF_img = centered_for_FF(zero_ped_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/4.35(c).jpg', centered_for_FF_img)

#4.35(d)
DFF_img = DFF(centered_for_FF_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/4.35(d).jpg', DFF_img)


#4.35(e)
gaussian_filter_img = gaussian_filter(50,centered_for_FF_img.shape)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/4.35(e).jpg', gaussian_filter_img)

#4.35(f)
elementwise_img = elementwise(DFF_img, gaussian_filter_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/4.35(f).jpg', elementwise_img)

#4.35(g)
IDFT_img = IDFT(elementwise_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/4.35(g).jpg', IDFT_img)
#4.35(h)
extracted_img = extract(origin_img,IDFT_img)
cv2.imwrite('C:/Users/Max/Desktop/python/programming/DIP/4.35(h).jpg', extracted_img)


