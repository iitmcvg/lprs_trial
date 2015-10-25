import cv2
import sys
import numpy as np

if len(sys.argv) < 3:
    print "Usage : " + sys.argv[0] + " <classifier> <image path>"
    sys.exit()

def getAdaptiveIndices(ar, loc, mAr, mLoc):
    # Make it adaptive
    l1, l2, a1, a2 = 0.6, 2.0, 0.6, 2.0
    count = len(np.where((ar > a1*mAr) & (ar < a2*mAr))[0])
    #while not count == 7 or count == 8:
    return np.where((ar > a1*mAr) & (ar < a2*mAr))[0]

cas = cv2.CascadeClassifier(sys.argv[1])

for img in sys.argv[2:]:
    name = img
    img = cv2.imread(img, 0)
    roi = []
    idx = 0
    for (a,b,c,d) in cas.detectMultiScale(img, 1.3, 2):
        if idx == 0:
            roi = [a, b, c, d]
            idx = idx + 1
        if c > roi[2] or d > roi[3]:
            roi = [a, b, c, d]
    if len(roi) == 4:
        roi = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        #cv2.imshow("roi", roi); cv2.waitKey(0);
        _, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow("roi", otsu);
        otsuBkup = otsu.copy()
        contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.array(contours)

        ar = []
        loc = []
        for cnt in contours:
            ar.append(cv2.contourArea(cnt))
            loc.append([np.mean(cnt[:, 0, 0]), np.mean(cnt[:, 0, 1])])

        ar = np.array(ar)
        loc = np.array(loc)
        mAr = np.mean(ar)
        mLoc = np.mean(loc)

        t = getAdaptiveIndices(ar, loc, mAr, mLoc)
        contoursFil = contours[t]

        roi_nos = []

        idx = 0
        for cnt in contoursFil:
            x, y, w, h = cv2.boundingRect(cnt)
            temp = otsuBkup[y:y+h, x:x+w]
            blkCount = len(np.where(temp == 0)[0])
            whiteCount = len(np.where(temp == 255)[0])
            # print blkCount, whiteCount
            # Compare number of black and white pixels
            cv2.imshow("roi_"+str(idx), temp);cv2.waitKey(0);
            roi_nos.append(temp)
            ++idx