import IPython
import cv2
import numpy as np
import time








for i in range(360):
	nom_img = cv2.imread("car_images/human.png",-1)
	nom_img = cv2.pyrDown(nom_img)
	size = nom_img.shape

	center = np.array([float(size[0])/2.0,float(size[1])/2.0])
	rot_mat = cv2.getRotationMatrix2D((center[0],center[1]),float(360-i), 1.0)

	rot_img = cv2.warpAffine(nom_img,rot_mat, (size[1],size[1]),flags=cv2.INTER_LINEAR) 
	cv2.imwrite("car_images/human_ "+str(i)+".png",rot_img)
	cv2.imshow("Fig "+str(i),rot_img)
	print i


IPython.embed()