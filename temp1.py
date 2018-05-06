#!/usr/bin/env python3
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# constants
IMAGE_LOCATION="/home/ujwal/Desktop/"

def retarea(blah):
	return blah[1]
def splice(in_imageROI,flag='b'):
	in_image=np.copy(in_imageROI)
	if flag=='b':
		in_image[:,:,1]=0;
		in_image[:,:,2]=0;
	if flag=='g':
		in_image[:,:,0]=0;
		in_image[:,:,2]=0;
	if flag=='r':
		in_image[:,:,0]=0;
		in_image[:,:,1]=0;
	return in_image	

def convert(in_image):
	b,g,r=cv2.split(in_image)
	imageROI=cv2.merge([r,g,b])
	return imageROI

def getimage(filename,flag=1):
	image=cv2.imread(os.path.join(IMAGE_LOCATION,filename),flag)
	return image

def main():
	x=7
	image=getimage("9.jpeg")
	image=cv2.copyMakeBorder(image,x,x,x,x,cv2.BORDER_CONSTANT,value=0)
	imageROIdisc=np.zeros(image.shape)
	imageROI,disc,disc1=cv2.split(imageROIdisc)
	disc2=np.copy(disc1)
	imageROIfinal=np.copy(imageROI)
	

	edge=cv2.Canny(image,50,400)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gray=np.float32(gray)
	

	dst = cv2.cornerHarris(gray,2,3,0.004)
	disc1[(dst>0.01*dst.max())]=255
	
	todo=[(0,0),(0,1)]
	todraw=[]
	threshold=0.4
	tc=0
	for i in range(0,image.shape[0]):
		for j in range(0,image.shape[1]):
			if  ((image[i][j][2]>image[i][j][1]) and (image[i][j][2]>(1.5*image[i][j][0])) and (image[i][j][1] in range(30,120)) ):
				imageROI[i][j]=255
				todo.append((i,j))
				print(str(i)+','+str(j))

	for i in range(len(todo)):
			tc=0
			for k in range(0,x):
				for l in range(0,x):
					if imageROI[todo[i][0]+k-(int(x/2))][todo[i][1]+l-(int(x/2))]==255:
						tc=tc+1
			print(tc)
			if (tc/(x*x))>threshold:
				for k in range(0,x):
					for l in range(0,x):
						imageROIfinal[todo[i][0]+k-(int(x/2))][todo[i][1]+l-(int(x/2))]=255					

	cv2.imshow('gtg',imageROIfinal)
	cv2.waitKey()
	cv2.destroyAllWindows()					
	for ila in range(0,image.shape[0]):
		for jila in range(0,image.shape[1]):
			if (imageROIfinal[ila][jila]==255):
				disc[ila][jila]=edge[ila][jila]
				disc2[ila][jila]=disc1[ila][jila]
				if disc1[ila][jila]==255:
					todraw.append((jila,ila))
				#image[ila][jila][2]=disc1[ila][jila]
	disc[disc2[:]>0]=255			
	cv2.imshow('dhu',edge)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	threshold=50
	
	for index in todraw:
		for jndex in todraw:
			if np.linalg.norm(np.array(index)-np.array(jndex))< threshold:
				cv2.line(disc2,index,jndex,255,1)
			#cv2.imshow('fgrw',disc2)
			#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	cv2.imwrite('rest.png',disc2)
	cv2.imwrite('reste.png',disc)
	discnew=getimage('rest.png')
	discnew[disc2[:]>0]=[0,0,255]
	template = getimage('cntTemp.png',0)
	img_rgb = getimage('rest.png')
	img_gray = getimage('rest.png',0)
	#container=np.zeros(img_rgb.shape)
	kernel=np.ones((5,5),np.uint8)
	img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
	img_gray=cv2.dilate(img_gray,kernel)
	img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
	i=0
	canderiuuuu=[]
	candidate=[]
	candidatelist=[]
	print("*****************************************")
	#template = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel)
	im2,contours,hierarchy = cv2.findContours(template,2,1)
	im2q,contoursq,hierarchyq = cv2.findContours(img_gray,2,1)
	#print(contours[0])
	#print("*****************************************")
	print(cv2.contourArea(contours[0]))
	for cnt in contoursq:
		ret = cv2.matchShapes(cnt,contours[0],1,0.0)
		#print(cnt)
		if (ret<0.5 ):	
			print("Match")	
			candidatelist.append((cnt,cv2.contourArea(cnt)))
			canderiuuuu.append(cv2.contourArea(cnt))

	img_rgb = getimage('reste.png')
	img_gray = getimage('reste.png',0)
	#container=np.zeros(img_rgb.shape)
	kernel=np.ones((5,5),np.uint8)
	img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
	img_gray=cv2.dilate(img_gray,kernel)
	img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
	i=0
	
	print("*****************************************")
	#template = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel)
	im2,contours,hierarchy = cv2.findContours(template,2,1)
	im2q,contoursq,hierarchyq = cv2.findContours(img_gray,2,1)
	print(cv2.contourArea(contours[0]))
	for cnt in contoursq:
		ret = cv2.matchShapes(cnt,contours[0],1,0.0)

		#print(cnt)
		if (ret<0.5 ):	
			print("Match")
			candidatelist.append((cnt,cv2.contourArea(cnt)))
			canderiuuuu.append(cv2.contourArea(cnt))
			
		#else:
	canderiv=[]
	candidate=sorted(candidatelist,key=retarea)	
	for i in range(0,(len(candidate)-1)):
		canderiv.append(candidate[i+1][1]-candidate[i][1])
	canderiuuuu.sort()
	plt.plot(canderiuuuu)
	plt.ylabel('Area')
	#plt.show()	
	upper=canderiv.index(max(canderiv))	
	del canderiv[upper:]
	lower=canderiv.index(max(canderiv))
	candidate1=candidate[(lower+1):(upper+1)]




	for cnt in candidate:
		rect=cv2.minAreaRect(cnt[0])
		box =cv2.boxPoints(rect)
		box =np.int0(box)
		cv2.drawContours(img_rgb,[box],0,(0,255,0),2)
		cv2.drawContours(image,[box],0,(0,255,0),2)
	
	cv2.imshow('TaDa',image)
	cv2.waitKey(0)
	cv2.imshow('fre',disc)
	cv2.waitKey(0)
	cv2.imshow('re',disc2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()					
	
	print("*****************************************")
	
	
	
if __name__ == '__main__':
	main()		