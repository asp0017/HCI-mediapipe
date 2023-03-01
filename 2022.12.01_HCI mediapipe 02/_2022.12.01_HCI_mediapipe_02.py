from datetime import datetime
import numpy as np
import cv2 as cv
import argparse
import mediapipe as mp

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                            OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

picPath = './GeneratedPictures/'


def selectFunc(index, inFrame):
    if index == 0:
        displayPicture()
    elif index == 1:
        playVideo()
    elif index == 2:
        savePicture()
    elif index == 3:
        takePicture(inFrame)

def displayPicture():
    capTmp = cv.imread('default_picture.png')
    cv.imshow('Default picture (type \'e\' to exit)', capTmp)
    while True:
        if cv.waitKey(0) == ord('e'):
            break
    cv.destroyWindow('Default picture (type \'e\' to exit)')
    print('Display picture.')
    return

def playVideo():
    capTmp = cv.VideoCapture('default_video.mp4')
    while True:
        retTmp, frameTmp = capTmp.read()
        if not retTmp:
            break
        frameTmp = cv.resize(frameTmp, (0, 0), fx=0.5, fy=0.5)
        cv.imshow('Default video (type \'e\' to exit)', frameTmp)
        if cv.waitKey(20) == ord('e'):
            break
    cv.destroyWindow('Default video (type \'e\' to exit)')
    print('Play video.')
    return

def savePicture():
    if not hasTakenPicture:
        print('No captured picture to save')
    else:
        cv.imwrite(pictureNameStr, takenPic)
        resetTakenPic()
        print('Save picture', pictureNameStr)
    return

def takePicture(inFrame):
    global hasTakenPicture
    hasTakenPicture = True
    global takenPic
    takenPic = inFrame
    timeNow = datetime.now()
    global pictureNameStr
    pictureNameStr = picPath + timeNow.strftime('pic_%Y%m%d_%H%M%S') + '.png'
    print('Take picture', pictureNameStr)
    return

def resetSumAndSumCounter(sum):
    for i in range(np.size(sum)):
        sum[i] = 0
        sumCounter[i] = 0

def resetTakenPic():
    takenPic[:] = (128, 128, 128)
    cv.putText( takenPic, 'NOT FOUND', 
                (20, 150), cv.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 3, cv.LINE_AA)
    global hasTakenPicture
    hasTakenPicture = False

# hands detection
class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image,draw=True):
        imageRGB = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
            if draw:
                cv.circle(image,(cx,cy), 15 , (255,0,255), cv.FILLED)

        return lmlist

    def pointsInArea(self,lmList,x1,y1,x2,y2):

        counter = 0

        if len(lmList) != 0:
            for id in range(len(lmList)):
                if x1 < lmList[id][1] and lmList[id][1] < x2 and y1 < lmList[id][2] and lmList[id][2] < y2:
                        counter += 1
        return counter


# execute
cap = cv.VideoCapture(0)
tracker = handTracker() # hand detection

pictureNameStr = picPath
hasTakenPicture = False
takenPic = np.zeros((300, 300, 3), np.uint8)
resetTakenPic()
# draw command areas
points = np.array( [[[500, 16], [620, 116]], 
                    [[500, 132], [620, 232]], 
                    [[500, 248], [620, 348]], 
                    [[500, 364], [620, 464]]])
commands = np.array(   ['DEFAULT PICTURE', 'DEFAULT VIDEO', 
                        'SAVE PICTURE', 'TAKE PICTURE'])
sum = np.zeros(np.size(points, 0), dtype=int)
sumCounter = np.zeros(np.size(sum), dtype=int)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    
    hd_results = tracker.handsFinder(frame)  # hand detection
    lmList = tracker.positionFinder(hd_results)

    cnt = np.zeros(np.size(points, 0), dtype=int)
    cntMax = -1
    cntMaxIndex = -1

    for i in range(np.size(points, 0)):
        # hand detection: 算 area[i] 裡有幾個點並更新最大值
        cnt[i] = tracker.pointsInArea(lmList, points[i][0][0], points[i][0][1], points[i][1][0], points[i][1][1], )
        if cnt[i] > cntMax:
            cntMax = cnt[i]
            cntMaxIndex = i

        # draw command area
        cv.rectangle(   frame,
                        (points[i][0][0], points[i][0][1]), 
                        (points[i][1][0], points[i][1][1]), 
                        (255, 255,255), 1)
        cv.putText( frame, commands[i], 
                    (points[i][0][0]+10, int((points[i][0][1]+points[i][1][1])/2)), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    0.4, (255, 255, 255), 1, cv.LINE_AA)

        sum[i] += cnt[i]
        if sum[i] > 10: # 在時間內累積的手部節點數超過定量時觸發指令
            selectFunc(i, frame)
            resetSumAndSumCounter(sum)
        sumCounter[i] += 1
        if(sumCounter[i] > 20): # 時間一到就重新算累積的手部節點數
            resetSumAndSumCounter(sum)
    
    
    cv.imshow("Hands Detection",hd_results) # hand detection

    if cv.waitKey(20) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
