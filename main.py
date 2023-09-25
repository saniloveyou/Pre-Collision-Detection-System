import os
import time
import cv2
import numpy as np
from threading import Thread
from control_panel import ControlPanel
from taskConditions import TaskConditions
from ObjectDetector.utils import ObjectModelType
from LaneDetector.laneDetector import LaneDetector
from ObjectDetector.yoloDetector import YoloDetector
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure
from LaneDetector.perspectiveTransformation import PerspectiveTransformation

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
cv2.setNumThreads(NUM_THREADS)


video_path, horizon, top_l, top_r, bottom_l, bottom_r, bottom = "./test_right_2.mov", 435, -80, 150, -300, 400, -80

object_config = {
	"model_path": './ObjectDetector/models/yolov8n.pt',
	"model_type" : ObjectModelType.YOLOV8,
	"classes_path" : './ObjectDetector/models/coco_label.txt',
	"box_score" : 0.1,
	"box_nms_iou" : 0.1
}

def detect_lane(frame):
    global pts, pts_left, pts_right, lane_infer_time
    lane_time = time.time()
    pts, pts_left, pts_right = laneDetector.process_frame(frame, False)
    lane_infer_time = round(time.time() - lane_time, 2)
    
def detect_object(frame):
    global obect_infer_time
    obect_time = time.time()
    objectDetector.DetectFrame(frame)
    obect_infer_time = round(time.time() - obect_time, 2)

#==========================================================
#	    				Initialize
#==========================================================
# Priority : FCWS > LDWS > LKAS

# lane detector
laneDetector = LaneDetector(horizon=horizon, left_x_offset=top_l, right_x_offset=top_r, bottom_l=bottom_l, bottom_r=bottom_r, bottom=bottom)
transformView = PerspectiveTransformation(src=np.floor(laneDetector.src*laneDetector.input_scale), dst=np.floor(laneDetector.dst*laneDetector.input_scale))

# object detection model
objectDetector = YoloDetector()
objectDetector.set_defaults(object_config)

# Distance Estimator
distanceDetector = SingleCamDistanceMeasure()

# display panel
displayPanel = ControlPanel()
analyzeMsg = TaskConditions()
count = 0
    

if __name__ == '__main__':
    # Initialize read video 
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = (1280, 960)
    cv2.namedWindow("ADAS Simulation", cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        
        ret, frame = cap.read() # Read frame from the video
        frame = cv2.resize(frame, (width, height))
        
        if count: count -= 1; continue
        else: count = 1
        
        if ret:	
            frame_show = frame.copy()
            
            #========================== Detect Model =========================
            t1 = Thread(target=detect_object, args=(frame,))
            t2 = Thread(target=detect_lane, args=(frame,))
            t1.start(), t2.start()
            t1.join(), t2.join()
    
            #========================= Analyze Status ========================
            distanceDetector.calcDistance(objectDetector.object_info)
            vehicle_distance = distanceDetector.calcCollisionPoint(pts*laneDetector.input_scale)
            analyzeMsg.UpdateCollisionStatus(vehicle_distance, True, distance_thres=10)
            
            birdview_show = transformView.transformToBirdView(frame_show, cv2.INTER_NEAREST)
            (vehicle_direction, vehicle_curvature) ,vehicle_offset, birdview_show = transformView.calcCurveAndOffset(birdview_show, pts_left*laneDetector.input_scale, pts_right*laneDetector.input_scale)        
            analyzeMsg.UpdateOffsetStatus(vehicle_offset, offset_thres=0.6)
            analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)
            
            #========================== Draw Results =========================
            birdview_show = transformView.DrawDetectedOnBirdView(birdview_show, pts*laneDetector.input_scale, analyzeMsg.offset_msg)
            # birdview_show = transformView.DrawTransformFrontalViewAreaOnBirdView(birdview_show) # for debug
            # frame_show = transformView.DrawTransformFrontalViewArea(frame_show) # for debug
            frame_show = laneDetector.DrawAreaOnFrame(frame_show, vehicle_offset)
            frame_show = objectDetector.DrawDetectedOnFrame(frame_show, vehicle_distance)
            frame_show = distanceDetector.DrawDetectedOnFrame(frame_show)
            
            frame_show = displayPanel.DisplayBirdViewPanel(frame_show, birdview_show)
            frame_show = displayPanel.DisplaySignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)	
            frame_show = displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg, obect_infer_time, lane_infer_time )
            cv2.imshow("ADAS Simulation", frame_show)
        else: break
        if cv2.waitKey(1) == ord('q'): break # Press key q to stop
    cap.release()
    cv2.destroyAllWindows()

