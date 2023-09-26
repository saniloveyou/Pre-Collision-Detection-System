# Package importation
import numpy as np
import cv2 
from openpyxl import Workbook # Used for writing data into an Excel file
from ultralytics import YOLO
import calibration
import os 
from functools import wraps
from ultralytics import YOLO
import time
from threading import Thread
import threading
import Relspeed
from playsound import playsound
import http.client, urllib


cv2.setNumThreads(4)

kernel= np.ones((7,7),np.uint8)
window_size = 3
min_disp = 2
num_disp = 20-min_disp

class conf():
   

    def __init__(self):
        self.stereo =cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2)
        self.kernel= np.ones((3,3),np.uint8)
        self.stereoR=cv2.ximgproc.createRightMatcher(self.stereo) # Create another stereo for right this time
        self.lmbda = 80000
        self.sigma = 1.8
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo)
        self.wls_filter.setLambda(self.lmbda)
        self.wls_filter.setSigmaColor(self.sigma)
        self.previous_distance = None
        self.previous_time = None
        self.filteredImg = None
    

    def stereo_remap(self):
        # if stereoMap doesnt exist do calibration
        if not os.path.isfile('FYP_Final/stereoMap.xml'):
            calibration.start()
            print("calibration done")

        cv_file = cv2.FileStorage()
        cv_file.open('FYP_Final/stereoMap.xml', cv2.FileStorage_READ)

        self.stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
        self.stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
        self.stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
        self.stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
        cv_file.release()

    def distance(self,x,y):
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += self.filteredImg[y+u,x+v]

        disparity=average/9
        focal_length = 4.4
        baseline = 4.1

        self.Distance = (round((1 -  (baseline * focal_length)/disparity ** 4, 3)))

        return self.Distance  

    def disparity(self,Right_corrected,Left_corrected):
        scale = 4
        width = int(640/scale)
        height = int(480/scale)

        Right_corrected = cv2.resize(Right_corrected, (width, height))
        Left_corrected = cv2.resize(Left_corrected, (width, height))

        grayR= cv2.cvtColor(Right_corrected,cv2.COLOR_BGR2GRAY)
        grayL= cv2.cvtColor(Left_corrected,cv2.COLOR_BGR2GRAY)


        def filterit():
            self.disp= self.stereo.compute(grayL,grayR)#.astypcoe(np.float32)/ 16
            dispL= self.disp
            dispR= self.stereoR.compute(grayR,grayL)
            dispL= np.int16(dispL)
            dispR= np.int16(dispR)
            self.filteredImg= self.wls_filter.filter(dispL,grayL,None,dispR)
            self.filteredImg = cv2.normalize(src=self.filteredImg, dst=self.filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            self.filteredImg = np.uint8(self.filteredImg)
            self.filteredImg = cv2.resize(self.filteredImg, (int(640/2), int(480/2)))
            
        t1 = threading.Thread(target=filterit, name="Thread2")
        t1.start()
        t1.join()
        cv2.imshow('Disparity Map', self.filteredImg)

class App:

    img_normal = cv2.imread("FYP_Final/assets/FCWS-normal.png",cv2.IMREAD_UNCHANGED)
    img_warning = cv2.imread("FYP_Final/assets/FCWS-warning.png",cv2.IMREAD_UNCHANGED)
    img_normal = cv2.resize(img_normal, (100, 100))
    img_warning = cv2.resize(img_warning, (100, 100))
    
    storedistance = 0


    configuration = conf()

    # left_camera_source = "data/Data/FullLeft.mov"
    # right_camera_source = "data/Data/FullRight.mov"

    left_camera_source = "data/left/left_trim.mp4"
    right_camera_source = "data/right/right_trim.mp4"

    Lframe = cv2.VideoCapture(left_camera_source)
    Rframe =  cv2.VideoCapture(right_camera_source)
    model = YOLO('yolov8n.pt') 
    model.fuse()

    configuration.stereo_remap()

    def get_frame(self):
        def getleft(): 
            self.left_frame = self.Lframe.read()[1]
            self.left_frame = cv2.resize(self.left_frame, (int(1920), int(960)))
        def getright(): 
            self.right_frame = self.Rframe.read()[1]
            self.right_frame = cv2.resize(self.right_frame, (int(1920), int(960)))
            # self.right_frame = cv2.flip(self.right_frame,-1)
        t1 = Thread(target=getleft)
        t2 = Thread(target=getright)
        t1.start(), t2.start(); t1.join(), t2.join()
        

    def DisplayCollisionPanel(self, main_show,prevdistance,apx_distance,mid_x,speed_fram_prev,speed_frame,x,y,point_x,point_y,show_ratio=0.25) :
        W = int(main_show.shape[1]* show_ratio) 
        H = int(main_show.shape[0]* show_ratio)
        widget = np.copy(main_show[H+20:2*H, -W-20:])
        widget //= 2
        widget[0:3,:] = [0, 0, 255]  # top 
        widget[-3:-1,:] = [0, 0, 255] # bottom 
        widget[:,-3:-1] = [0, 0, 255] #left 
        widget[:,0:3] = [0, 0, 255]  # right 
        main_show[H+20:2*H, -W-20:] = widget
        y ,x = self.img_warning[:,:,3].nonzero()

        distance = "Distance: " + str(apx_distance)

        cv2.putText(main_show, distance,(1060, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        # print(apx_distance,mid_x,)

        if apx_distance <=15 and mid_x > 0.3 and mid_x < 0.7 and prevdistance > apx_distance:

            flow = Relspeed.calculate_optical_flow(speed_fram_prev,speed_frame)
            speed = Relspeed.calculate_relative_speed(flow)
            speed = round(speed,3)
            cv2.putText(main_show, "Speed: "+ str(speed),(1060, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if speed > 4:
                Relspeed.play()       
                main_show[H+y+50, (x-W-5)] = self.img_warning[y, x, :3]
                cv2.putText(main_show, "Warning Risk", (1060, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            else:
                main_show[H+y+50, (x-W-5)] = self.img_normal[y, x, :3]
                cv2.putText(main_show, "Normal Risk", (1060, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        else:
            main_show[H+y+50, (x-W-5)] = self.img_normal[y, x, :3]
            cv2.putText(main_show, "Normal Risk", (1060, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
       
        return main_show  



    def run(self):
        flag = False
        ret, speed_fram_prev = self.Rframe.read()
        self.speed_frame = speed_fram_prev.copy()
        frame_width = 640
        frame_height = 480
        self.prevdistance = 0

        # create a track bar for the right plot 0 to 

        x = [0, frame_width-190, round((frame_width/2)+60), round((frame_width/2)-60)]
        y = [frame_height, frame_height-1, round(frame_height/2), round(frame_height/2)]
        
        cv2.namedWindow("Img 2", cv2.WINDOW_NORMAL)
        # change the y[2] using trackbar 
        def nothing(x):
            pass
        cv2.createTrackbar('y', 'Img 2', y[2], 480, nothing)
        cv2.createTrackbar('x right', 'Img 2', x[2], 640, nothing)
        cv2.createTrackbar('x left', 'Img 2', x[3], 640, nothing)

        count  = 0

        

        while(True):

            if count:
                count = 0
                continue
            else: count = 1

            self.get_frame()

            start = time.time()

            x[2] = cv2.getTrackbarPos('x right', 'Img 2')
            x[3] = cv2.getTrackbarPos('x left', 'Img 2')
            y[2] = cv2.getTrackbarPos('y', 'Img 2')
            
            self.Left_corrected= cv2.remap(self.left_frame,self.configuration.stereoMapL_x,self.configuration.stereoMapL_y, interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)  # Rectify the image using the kalibration parameters founds during the initialisation
            self.Right_corrected= cv2.remap(self.right_frame,self.configuration.stereoMapR_x,self.configuration.stereoMapR_y, interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)

         
            self.Right_corrected = self.right_frame
            self.Left_corrected = self.left_frame


            self.Left_corrected = cv2.resize(self.Left_corrected, (int(640), int(480)))
            self.Right_corrected = cv2.resize(self.Right_corrected, (int(640), int(480)))

            rights = self.model(self.Right_corrected, show=False,imgsz=320, verbose=False,conf=0.2)
            lefts = self.model(self.Left_corrected, show=False,imgsz=320, verbose=False,conf=0.2)
            
            height, width = self.Right_corrected.shape[:2]

            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime

            if (lefts[0].__len__() == 0  or rights[0].__len__() == 0):
                cv2.putText(self.Right_corrected, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.imshow('Img 2',self.Right_corrected)
            

            if (lefts[0].__len__() >=1 and rights[0].__len__()>=1):

                for (left, right) in zip(lefts, rights):
                    right_plot = right.plot()
                    
                    right_x1, right_y1, right_x2, right_y2 = right.boxes.xyxy[0].cpu().numpy()
                    left_x1, left_y1, left_x2, left_y2 = left.boxes.xyxy[0].cpu().numpy()

                    mid_x = ((right_x1/width + right_x2/width) / 2)
                    mid_y = ((right_y1/height + right_y2/height) / 2)

                    left_points = [int(left_x1), int(left_y1), int(left_x2), int(left_y2)]
                    right_points = [int(right_x1), int(right_y1), int(right_x2), int(right_y2)]
  

                    mid_pointx = int((right_x1 + right_x2) / 2)
                    mid_pointy = int((right_y1 + right_y2) / 2)

                    cv2.line(right_plot, (x[0],y[0]),(x[1],y[1]),(0,255,0),2)
                    cv2.line(right_plot, (x[1],y[1]),(x[2],y[2]),(0,255,0),2)
                    cv2.line(right_plot, (x[2],y[2]),(x[3],y[2]),(0,255,0),2)
                    cv2.line(right_plot, (x[3],y[2]),(x[0],y[0]),(0,255,0),2)

                    # self.Right_corrected[
                    #         right_points[1]:right_points[3],
                    #         right_points[0]:right_points[3]
                    #         ] = (0, 255, 0)

                    # self.Left_corrected[
                    # left_points[1]:left_points[3],
                    # left_points[0]:left_points[3]
                    # ] = (0, 255, 0)

                    # point_x = int((right_x1 + right_x2) // 2)
                    # point_y = int(right_y2)
                    point_x = int(mid_pointx)
                    point_y = int(mid_pointy)



                    self.speed_frame = self.Right_corrected.copy()
                    # if the right_x and right_y boxes are in the box of x[0],x[1],x[2],x[3] and y[0],y[1],y[2],y[3], save the boxes
                    if point_x > x[0] and point_x < x[1] and point_x < x[2] and point_x > x[3]:
                        if point_y < y[0] and point_y < y[1] and point_y > y[2] and point_y > y[2]:
                            cv2.circle(right_plot, (point_x, point_y), 5, (0, 0, 255), -1)


                # run after 10 second 
                if self.configuration.previous_time is None:
                    self.configuration.previous_time = time.time()
                elif time.time() - self.configuration.previous_time > 15:
                    self.configuration.previous_time = time.time()
                    speed_fram_prev = self.speed_frame
                    speed_fram_prev = speed_fram_prev[int(right_y1-120):int(right_y2+120), int(right_x1-20):int(right_x2+20)]

                # compute speed 
                speed_frame = self.speed_frame.copy()
    
                if flag:
                    # better conf
                    speed_frame = speed_frame[int(right_y1-120):int(right_y2+120), int(right_x1-20):int(right_x2+20)]

                    # speed_frame = speed_frame[int(right_y1-20):int(right_y2+20), int(right_x1-20):int(right_x2+20)]

                    # speed_frame = speed_frame[int(right_y1):int(right_y2), int(right_x1):int(right_x2)]

                    try:
                        speed_fram_prev = cv2.resize(speed_fram_prev, (640, 480))
                        speed_frame = cv2.resize(speed_frame,  (640, 480))
                    except:
                        pass
                try:
                    speed_fram_prev = cv2.resize(speed_fram_prev,  (640, 480))
                    speed_frame = cv2.resize(speed_frame, (640, 480))
                except:
                    pass

                # concate two frame left and right
                
                # cv2.imshow("previos", speed_fram_prev)
                # cv2.imshow("current", speed_frame)


                right_plot = cv2.resize(right_plot, (int(1280), int(960)))

                # compute disparity
                self.configuration.disparity(self.Right_corrected,self.Left_corrected)
                apx_distance = self.configuration.distance(int(mid_pointx/4),int(mid_pointy/4))
                apx_distance = round(apx_distance,3)
                apx_distance = (apx_distance*100)/2.5

  
                if flag:
                    right_plot = self.DisplayCollisionPanel(right_plot,self.prevdistance,apx_distance,mid_x,speed_fram_prev,speed_frame,x,y,point_x,point_y)
                    self.prevdistance = apx_distance
                flag = True
               

                # compute fps
                end = time.time()
                totalTime = end - start
                fps = 1 / totalTime
                
                cv2.putText(right_plot, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
                cv2.imshow('Img 2', right_plot)  
                cv2.imshow('Both Images', np.hstack([self.Left_corrected, self.Right_corrected]))  


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.Rframe.release()
        self.Lframe.release()
        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    app = App()
    app.run()

    


