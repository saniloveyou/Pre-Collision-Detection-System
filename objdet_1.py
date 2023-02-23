import os

# print(os.getcwd())
os.chdir("C:/Users/Admin/Documents/objdet_project")

import cv2
import numpy as np
import time

np.random.seed(42)

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # initialize the model
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, "__Background__")

        self.colorList = np.random.uniform(low=0, high=256, size=(len(self.classesList), 3))
        # print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if (cap.isOpened() == False):
            print("Error reading the video file")
            return
        
        (success, image) = cap.read()
        
        startTime = 0
        while success:
            currentTime = time.time()
            fps = 1/(currentTime-startTime)
            startTime = currentTime

            classLabelIDs, confidences, bbox = self.net.detect(image, confThreshold=0.5)

            bbox = list(bbox)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bbox, confidences, score_threshold = 0.5, nms_threshold=0.2)

            if len(bboxIdx) != 0:
                for i in range(len(bboxIdx)):

                    bb = bbox[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}: {:.4f}".format(classLabel, classConfidence)

                    x, y, w, h = bb
                    cv2.rectangle(image, (x, y), (x+w, y+h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                    ## draw line
                    lineWidth = int(w*0.3)
                    cv2.line(image, (x, y), (x+lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x, y), (x, y+lineWidth), classColor, thickness=5)
            cv2.putText(image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            (success, image) = cap.read()
        cv2.destroyAllWindows()

