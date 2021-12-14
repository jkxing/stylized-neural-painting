import cv2,sys
import numpy as np
img = cv2.imread(sys.argv[1])
print(img.shape)
x = np.zeros(img.shape[:2],dtype = np.uint8)
h,w = x.shape
x[h//6:h//6*5,w//6:w//6*5] = 1
x[h//8*3:h//8*5,w//8*3:w//8*5] = 2
print("_mask.".join(sys.argv[1].split(".")))
cv2.imwrite("_mask.".join(sys.argv[1].split(".")),x)