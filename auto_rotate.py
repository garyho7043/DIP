import numpy as np
import cv2

  


def rotate_bound(image, angle):
    
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    cos = np.abs(M[0, 0])

    sin = np.abs(M[0, 1])

  

 

    nW = int((h * sin) + (w * cos))

#     nH = int((h * cos) + (w * sin))

    nH = h

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

  

    return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

  


def get_minAreaRect(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255,

        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))

    return cv2.minAreaRect(coords)

  
  
  
filename = "./DIP/107422.jpg"
image = cv2.imread(filename)



angle = get_minAreaRect(image)[-1]
rotated = rotate_bound(image, angle)

  

cv2.putText(rotated, "angle: {:.2f} ".format(angle),
    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

  



print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("imput", image)
cv2.imshow("output", rotated)
cv2.waitKey(0)



