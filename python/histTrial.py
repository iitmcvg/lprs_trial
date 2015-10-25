ls
myp
%myp
cd Desktop/myproj/haar\ py\ tests
ls
run classifier.py haarcascade_russian_plate_number.xml images/2715DTZ.jpg
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
roi
cv2.calcHist([roi], [0], None, [256], [0, 256])
hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
from matplotlib import pyplot as plt
plt.hist(roi.ravel(), 256, [0, 256]); plt.show();
_, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("roi", otsu);cv2.waitKey(0);
cv2.imshow("roi", otsu);cv2.waitKey(0);
otsu_bkup = otsu
cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
im2, contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(otsu, contours, -1, (0, 255, 0), 3)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("roi", otsu);cv2.waitKey(0);
import numpy as np
mask = np.zeros(otsu.shape, np.uint8)
for cnt in contours:
    if 200<cv2.contourArea(cn
    
    
    
    exit
    
    
    
    ;
    X):
        vdk
cv2.imshow("roi", otsu);cv2.waitKey(0);
cv2.imshow("roi", otsu_bkup);cv2.waitKey(0);
_, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("roi", otsu);cv2.waitKey(0);
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if 200<cv2.contourArea(cnt)<5000:
        cv2.drawContours(img, [cnt], 0, 255, 2)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
cv2.imshow("roi", img);cv2.waitKey(0);
cv2.imshow("roi", img);cv2.waitKey(0);
cv2.imshow("roi", mask);cv2.waitKey(0);
cv2.imshow("roi", mask);cv2.waitKey(0);
contours
ar = []
for cnt in contours:
    ar.append(cv2.contourArea(cnt))
ar
plt.hist(ar, 256, [0, 256]); plt.show();
np.mean(ar)
m = np.mean(ar)
np.where(ar>0.8*m and ar<1.2*m)
np.where(ar>0.8*m)
np.where(ar>0.8*m && ar < 1.2*m)
np.where(ar>0.8*m and ar < 1.2*m)
np.where(ar>0.8*m & ar < 1.2*m)
np.where((ar>0.8*m) & (ar < 1.2*m))
np.where((ar>0.6*m) & (ar < 1.6*m))
np.where((ar>0.6*m) & (ar < 3*m))
ar[np.where((ar>0.6*m) & (ar < 3*m))]
ar[np.where((ar>0.6*m) & (ar < 3*m))[0]]
ar = np.array(ar)
ar[np.where((ar>0.6*m) & (ar < 3*m))[0]]
t=np.where((ar>0.6*m) & (ar < 3*m))
contours[t]
t=np.where((ar>0.6*m) & (ar < 3*m))[0]
contours[t]
contours = np.array(contours)
contours[t]
contoursFil = contours[t]
cv2.drawContours(otsu, contoursFil, -1, (0, 255, 0), 3)
cv2.drawContours(otsu, contoursFil, -1, (0, 255, 0), 3)
cv2.imshow("roi", otsu);cv2.waitKey(0);
img2 = np.zeros(mask.shape, dtype=np.uint8)
cv2.drawContours(img2, contoursFil, -1, (0, 255, 0), 3)
cv2.imshow("roi", img2);cv2.waitKey(0);
cv2.drawContours(img2, contoursFil, -1, 255, 3)
cv2.imshow("roi", img2);cv2.waitKey(0);
t=np.where((ar>0.6*m) & (ar < 2*m))[0]
len(t)
cv2.drawContours(img2, contours[t], -1, 255, 3)
cv2.imshow("roi", img2);cv2.waitKey(0);
img2 = np.zeros(mask.shape, dtype=np.uint8)
cv2.drawContours(img2, contours[t], -1, 255, 3)
cv2.imshow("roi", img2);cv2.waitKey(0);
%history
%history > hist.py
ls
export('history', 'hist.py')
import sys
export('history', 'hist.py')
%history -f hist.py
ls
cv2.imshow("roi", img2);cv2.waitKey(0);
cnt
cv2.moments(cnt)
np.mean(cnt)
np.mean(cnt[0,:])
cnt
np.mean(cnt[:, 0])
np.mean(cnt[:, 1])
np.mean(cnt[:, ])
cnt[0]
cnt[:,0]
cnt[:,1]
cnt[:][0]
cnt[:][1]
cnt[0]
cnt[1]
cnt[1][0]
cnt[1][0][0]
cnt[:][0][0]
cnt[2][0][0]
cnt[3][0][0]
cnt
cnt[:]
cnt[:,0,0]
np.mean(cnt[:,0,0])
cnt[:,0,1]
cv2.kmeans(otsu, 2)
cv2.kmeans(otsu, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER), 10, 1.0)
cv2.kmeans(otsu, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10,cv2.KMEANS_RANDOM_CENTERS)
cv2.kmeans(otsu, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10,cv2.KMEANS_RANDOM_CENTERS)
cv2.KMEANS_RANDOM_CENTERS
cv2.kmeans(otsu, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10,int(cv2.KMEANS_RANDOM_CENTERS))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
cv2.kmeans(np.float32(otsu), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
cv2.kmeans(np.float32(otsu), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
ret, label, center  = cv2.kmeans(np.float32(otsu), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
ret, label, center  = cv2.kmeans(np.float32(otsu), 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
label
ret, label, center  = cv2.kmeans(np.float32(otsu), 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
otsu == 255
np.where(otsu == 255)
np.where(otsu == 0)
otsu
cnt[1][0]
_, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
np.where(otsu == 0)
zip(np.where(otsu == 0))
zip(np.where(otsu == 0)[0], n)
t = np.where(otsu == 255)
zip(t[0], t[1])
ret, label, center  = cv2.kmeans(np.float32(zip(t[0], t[1]), 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


)
points = zip(t[0], t[1])
ret, label, center  = cv2.kmeans(np.float32(zip(t[0], t[1]), 2, criteria,



 10, cv2.KMEANS_RANDOM_CENTERS)



)
ret, label, center  = cv2.kmeans(np.float32(points), 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
ret
label
a = otsu(label.ravel() == 0)
a = otsu[label.ravel() == 0]
a = points[label.ravel() == 0]
points = np.float32(points)
a = points[label.ravel() == 0]
b = points[label.ravel() == 1]
A = a
B = b
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
ret, label, center  = cv2.kmeans(np.float32(zip(t[1], t[0]), 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

)
ret, label, center  = cv2.kmeans(np.float32(zip(t[1], t[0])), 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
A = points[label.ravel() == 0]
points = np.float32(zip(t[1], t[0]))
ret, label, center  = cv2.kmeans(points, 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
A = points[label.ravel() == 0]
B = points[label.ravel == 1]
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
B
A
points = np.float32(zip(t[0], t[1))
;)
points = np.vstack(t[0], t[1])
points = np.vstack(t)
points
points = np.float32(points)
cv2.kmeans(np.float32(otsu), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
ret, label, center  = cv2.kmeans(points, 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
A =
A = points[label.ravel() == 0]
B = points[label.ravel == 1]
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
B
A
points
points = np.vstack((t[1], t[0]))
ret, label, center  = cv2.kmeans(points, 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
points = np.float32(points)
ret, label, center  = cv2.kmeans(points, 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
A = points[label.ravel() == 0]
B = points[label.ravel == 1]
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
A
B
points
np.vstack
points = zip(t[1], t[0])
ret, label, center  = cv2.kmeans(points, 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
points = np.float32(points)
ret, label, center  = cv2.kmeans(points, 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
A
B
A = points[label.ravel() == 0]
B = points[label.ravel == 1]
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
label
A
B
points
label == 0
A = points[label.ravel() == 0]
A
B = points[label.ravel() == 1]
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
ret, label, center  = cv2.kmeans(points, 3, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
A = points[label.ravel() == 0]
B = points[label.ravel() == 1]
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
ret, label, center  = cv2.kmeans(points, 10, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
A = points[label.ravel() == 0]
B = points[label.ravel() == 1]
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
r
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
roi.copy()
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/7215BGN.JPG
run classifier.py haarcascade_russian_plate_number.xml images/9773BNB.jpg
run classifier.py haarcascade_russian_plate_number.xml images/9588DWV.jpg
run classifier.py haarcascade_russian_plate_number.xml images/3028BYS.JPG
run classifier.py haarcascade_russian_plate_number.xml images/3028BYS.JPG
run classifier.py haarcascade_russian_plate_number.xml images/3028BYS.JPG
run classifier.py haarcascade_russian_plate_number.xml images/3028BYS.JPG
run classifier.py haarcascade_russian_plate_number.xml images/3028BYS.JPG
run classifier.py haarcascade_russian_plate_number.xml images/9588DWV.jpg
run classifier.py haarcascade_russian_plate_number.xml images/9588DWV.jpg
run classifier.py haarcascade_russian_plate_number.xml images/3028BYS.JPG
run classifier.py haarcascade_russian_plate_number.xml images/3028BYS.JPG
cls
ls
cd ..
cd lprs_trial/
ls
run lprs.py xmls/haarcascade_russian_plate_number.xml images/3028BYS.JPG
run lprs.py xmls/haarcascade_russian_plate_number.xml images/*
xport('history', 'hist.py')
run lprs.py xmls/haarcascade_russian_plate_number.xml images/*
run lprs.py xmls/haarcascade_russian_plate_number.xml images/*
run lprs.py xmls/haarcascade_russian_plate_number.xml images/*
mva
0.86/1.62
run lprs.py xmls/haarcascade_russian_plate_number.xml images/*
mva
run lprs.py xmls/haarcascade_russian_plate_number.xml images/*
mva
np.mean(mva)
run lprs.py xmls/haarcascade_russian_plate_number.xml images/*
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run svm.py
run svm.py
svm.predict(testData[0])
testData[0]
ls
cv2.imshow("roi", img);cv2.waitKey(0);
run svm.py
cv2.imshow("roi", img);cv2.waitKey(0);
img
img==255
img==1
np.where(img == 255)
np.where(img == 1)
np.where(img == 3)
np.where(img == 0)
np.where(img == 256)
img.shape
i
len(cells)
cells[0]
len(cells[0])
len(cells[0][0])
len(cells[0][0][0])
len(cells[0][0][0][0])
cv2.resize(temp, (20, 20))
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
rm -r 0.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
svm.predict(temp)
temp
svm.predict(temp)
cells
cells[0]
cells[0][0]
cells[0][0][0]
len(cells[0][0][0])
svm
svm
svm.predict(temp)
temp
cv2.imwrite("output/temp.jpg", temp)
import pytesseract
pytesseract.image_to_string(temp)
pytesseract.image_to_string(temp)
t1 = cv2.imread("output/temp.jpg",0)
pytesseract.image_to_string(t1)
t1 = cv2.imread("output/temp.jpg")
pytesseract.image_to_string(t1)
t1 = cv2.imread("output/temp.jpg")
t1
from PIL import Image
pytesseract.image_to_string(Image.open("output/temp.jpg"))
pytesseract.image_to_string(Image.open("output/1.jpg"))
pytesseract.image_to_string(Image.open("output/2.jpg"))
cv2.imwrite("output/temp.jpg", temp)
cv2.imshow("roi", temp);cv2.waitKey(0);
cv2.imshow("roi", temp);cv2.waitKey(0);
cv2.imshow("roi", temp);cv2.waitKey(0);
t1
cv2.imshow("roi", t1);cv2.waitKey(0);
cv2.imwrite("output/temp.jpg", otsu)
pytesseract.image_to_string(Image.open("output/temp.jpg"))
cv2.imshow("roi", otsu);cv2.waitKey(0);
cv2.imwrite("output/temp.jpg", otsuBkup)
cv2.imshow("roi", otsuBkup);cv2.waitKey(0);
temp
t1
cv2.imshow("roi", t1);cv2.waitKey(0);
kernel = np.ones((5, 5), np.uint8)
er = cv2.erode(img, kernel, iterations=1)
cv2.imshow("roi", er);cv2.waitKey(0);
er = cv2.erode(t1, kernel, iterations=1)
cv2.imshow("roi", er);cv2.waitKey(0);
er = cv2.erode(t1, np.ones((3, 3), np.uint8), iterations=1)
cv2.imshow("roi", er);cv2.waitKey(0);
run lprs.py xmls/haarcascade_russian_plate_number.xml images/
cv2.imshow("roi_"+str(idx), temp);cv2.waitKey(0);
er = cv2.erode(t1, np.ones((3, 3), np.uint8), iterations=1)
run lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run python/lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
run python/lprs.py xmls/haarcascade_russian_plate_number.xml images/9588DWV.jpg
hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
%history -f python/histTrial.py
