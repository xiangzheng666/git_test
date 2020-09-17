#import new
import cv2
import numpy as np

path=r'test.jpg'

img=cv2.imread(path,1)
a=np.array(img)
print(a)
a.astype(np.uint8)
print(a)
#img=new.xiwei(img)

cv2.imshow('a',a)
cv2.waitKey(100)