import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

for i in range(5000,5001):

#edge detection
 image=cv2.imread('/home/ghosh/Documents/ROC/dataset'+str(i)+'.jpg')
 img = cv2.imread('/home/ghosh/Documents/ROC/label'+str(i)+'.jpg',0)
 image=cv2.resize(image,(480,240))
 img=cv2.resize(img,(480,240))
 cv2.imshow("image",image)
 blur = cv2.GaussianBlur(img,(5,5),0)
 sobel = cv2.Sobel(blur,cv2.CV_64F,1,1,ksize=3)
 dilation = cv2.dilate(sobel,(5,5),iterations = 1)
 erosion = cv2.erode(dilation,(5,5),iterations = 2)

 #cv2.imshow("edge",erosion)

#perspective transform
 roi=erosion[120:240,0:480]
 #cv2.imshow("roi",roi)
 
 pts1 = np.float32([[10,120],[450,120],[0,0],[480,0]])
 pts2 = np.float32([[175,120],[230,120],[0,0],[480,0]])

 M = cv2.getPerspectiveTransform(pts1,pts2)

 dst = cv2.warpPerspective(roi,M,(480,120))
 cv2.imshow("transform",dst)

#


 plt.subplot(221),plt.imshow(img),plt.title('Input')
 plt.subplot(222),plt.imshow(erosion),plt.title('Edge Detection')
 plt.subplot(223),plt.imshow(roi),plt.title('Roi') 
 plt.subplot(224),plt.imshow(dst),plt.title('Perspective Transform') 
 plt.show()

 cv2.waitKey(0)
