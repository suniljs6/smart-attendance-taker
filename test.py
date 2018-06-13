import cv2
import Image
import numpy as np
def mse(imageA,imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

face_cascade = cv2.CascadeClassifier('/root/Downloads/project/haarcascade_frontalface_alt.xml')
img = cv2.imread('input.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print (len(faces))
found = 0
count=1
for(x,y,w,h) in faces:
	crop = img[y:y+h,x:x+w]
	resize_crop = cv2.resize(crop,(108,108))
	for i in range(2,65):
		image = cv2.imread(str(i)+".jpg")
		error = mse(resize_crop,image)
		print error
		if(error<20):
			found+=1
			break
print found
cv2.waitKey(0)
cv2.destroyAllWindows()
