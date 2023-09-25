import os
import cv2
import sys
import torch
import random
import logging
import numpy as np

from ultralytics import YOLO


try: from utils import ObjectModelType, hex_to_rgb
except: from ObjectDetector.utils import ObjectModelType, hex_to_rgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available(): device = torch.device("mps")
print(f'Using {device} for inference')

class YoloDetector(object):
	_defaults = {
		"model_path": '/Users/ChengBoon/Documents/GitHub/FYP_Final/ObjectDetector/models/yolov8n.pt',
		"model_type" : ObjectModelType.YOLOV8,
		"classes_path" : '/Users/ChengBoon/Documents/GitHub/FYP_Final/ObjectDetector/models/coco_label.txt',
		"box_score" : 0.1,
		"box_nms_iou" : 0.1
	}

	@classmethod
	def set_defaults(cls, config): cls._defaults = config

	@classmethod
	def check_defaults(cls): return cls._defaults
		
	@classmethod
	def get_defaults(cls, n):
		if n in cls._defaults: return cls._defaults[n]
		else: return "Unrecognized attribute name '" + n + "'"

	def __init__(self, **kwargs):
		self.__dict__.update(self._defaults) # set up default values
		self.__dict__.update(kwargs) # and update with user overrides
		self.keep_ratio = False
		self.lite =  False
		self.input_shapes = (1280, 960)
		
		classes_path = os.path.expanduser(self.classes_path)
		if (os.path.isfile(classes_path) is False): raise Exception("%s is not exist." % classes_path)
		self._get_class(classes_path)

		model_path = os.path.expanduser(self.model_path)
		if (os.path.isfile(model_path) is False): raise Exception("%s is not exist." % model_path)
		
		self.model = YOLO(model_path)
		self.model.fuse()
		self.model.to(device)
		

	def _get_class(self, classes_path):
		with open(classes_path) as f: class_names = f.readlines()
		self.class_names = [c.strip() for c in class_names]
		get_colors = list(map(lambda i:"#" +"%06x" % random.randint(0, 0xFFFFFF),range(len(self.class_names)) ))
		self.colors_dict = dict(zip(list(self.class_names), get_colors))

	def adjust_boxes_ratio(self, bounding_box, ratio, stretch_type) :
		""" Adjust the aspect ratio of the box according to the orientation """
		xmin, ymin, width, height = bounding_box 
		width = int(width)
		height = int(height)
		xmax = xmin + width
		ymax = ymin + height
		if (ratio != None): ratio = float(ratio)
		return (xmin, ymin, xmax, ymax)

	def get_kpss_coordinate(self,kpss) :
		if (kpss != []): kpss = np.vstack(kpss)
		return kpss

	def get_boxes_coordinate(self, bounding_boxes) :
		if (bounding_boxes != []) :
			bounding_boxes = np.vstack(bounding_boxes)
			bounding_boxes[:, 2:4] = bounding_boxes[:, 2:4] - bounding_boxes[:, 0:2]
		return bounding_boxes

	def get_nms_results(self, bounding_boxes, confidences, class_ids, kpss, score, iou, priority=False):
		results = []
		nms_results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, score, iou) 
		if len(nms_results) > 0:
			for i in nms_results:
				kpsslist = []
				try :
					predicted_class = self.class_names[class_ids[i]]
				except :
					predicted_class = "unknown"
				if (kpss != []) :
					for j in range(5):
						kpsslist.append((int(kpss[i, j, 0]) , int(kpss[i, j, 1]) ) )
				
				bounding_box = bounding_boxes[i]
				bounding_box = self.adjust_boxes_ratio(bounding_box, None, None)

				xmin, ymin, xmax, ymax = list(map(int, bounding_box))
				results.append(([ymin, xmin, ymax, xmax, predicted_class], kpsslist))
		if (priority and len(results) > 0) :
			results = [results[0]]
		return results

	def cornerRect(self, img, bbox, t=5, rt=1, colorR=(255, 0, 255), colorC=(0, 255, 0)):
		ymin, xmin, ymax, xmax, label = bbox
		l = max(1, int(min( (ymax-ymin), (xmax-xmin))*0.2))

		if rt != 0:
			cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colorR, rt)
		# Top Left  xmin, ymin
		cv2.line(img,  (xmin, ymin), (xmin + l, ymin), colorC, t)
		cv2.line(img,  (xmin, ymin), (xmin, ymin + l), colorC, t)
		# Top Right  xmax, ymin
		cv2.line(img, (xmax, ymin), (xmax - l, ymin), colorC, t)
		cv2.line(img, (xmax, ymin), (xmax, ymin + l), colorC, t)
		# Bottom Left  xmin, ymax
		cv2.line(img, (xmin, ymax), (xmin + l, ymax), colorC, t)
		cv2.line(img, (xmin, ymax), (xmin, ymax - l), colorC, t)
		# Bottom Right  xmax, ymax
		cv2.line(img, (xmax, ymax), (xmax - l, ymax), colorC, t)
		cv2.line(img, (xmax, ymax), (xmax, ymax - l), colorC, t)

		return img

	def DetectFrame(self, srcimg) :
		kpss = []
		class_ids = []
		confidences = []
		bounding_boxes = []
		ids = []
		score = float(self.box_score)
		iou = float(self.box_nms_iou)
		
		output_from_network = self.model(srcimg, verbose=False, imgsz=320)

		# inference output
		for results in output_from_network:
			boxes = results.boxes
			for box in boxes:
				classId = int(box.cls)
				confidence = box.conf
				if confidence > score:
					x, y, w, h = int(box.xywh[0][0]), int(box.xywh[0][1]), int(box.xywh[0][2]), int(box.xywh[0][3])
					class_ids.append(classId)
					confidences.append(float(confidence))
					bounding_boxes.append(np.stack([(x - 0.5 * w), (y - 0.5 * h), (x + 0.5 * w), (y + 0.5 * h)], axis=-1))
			bounding_boxes = self.get_boxes_coordinate(bounding_boxes)
			kpss = self.get_kpss_coordinate(kpss)
			self.object_info = self.get_nms_results(bounding_boxes, confidences, class_ids, kpss, score, iou)

	def DrawDetectedOnFrame(self, frame_show, vehicle_distance=None) :
		tl = 3 or round(0.002 * (frame_show.shape[0] + frame_show.shape[1]) / 2) + 1    # line/font thickness
		if ( len(self.object_info) != 0 )  :
			for box, kpss in self.object_info:
				ymin, xmin, ymax, xmax, label = box
				if (len(kpss) != 0) :
					for kp in kpss :
						cv2.circle(frame_show,  kp, 1, (255, 255, 255), thickness=-1)
				c1, c2 = (xmin, ymin), (xmax, ymax)        
				tf = max(tl - 1, 1)  # font thickness
				t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
				c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

				obj_x = (xmax + xmin) // 2
				obj_y = ymin
						
				if (label != 'unknown'):
					if (vehicle_distance != None) :
						x,y,d = vehicle_distance
						if obj_x == x and obj_y == y:
							cv2.rectangle(frame_show, c1, c2, (0,0,255), -1, cv2.LINE_AA)
							self.cornerRect(frame_show, box, colorR=(0,0,255), colorC=(0,0,255))
						else:
							cv2.rectangle(frame_show, c1, c2, hex_to_rgb(self.colors_dict[label]), -1, cv2.LINE_AA)
							self.cornerRect(frame_show, box, colorR= hex_to_rgb(self.colors_dict[label]), colorC= hex_to_rgb(self.colors_dict[label]))
					else:
						cv2.rectangle(frame_show, c1, c2, hex_to_rgb(self.colors_dict[label]), -1, cv2.LINE_AA)
						self.cornerRect(frame_show, box, colorR= hex_to_rgb(self.colors_dict[label]), colorC= hex_to_rgb(self.colors_dict[label]))
				else :
					cv2.rectangle(frame_show, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)
					self.cornerRect(frame_show, box, colorR= (0, 0, 0), colorC= (0, 0, 0) )
				cv2.putText(frame_show, f'{label}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, (255, 255, 255), 2)
		return frame_show
		


if __name__ == "__main__":
	import time
	import sys

	capture = cv2.VideoCapture('test_right_2.mov')
	config = {
		"model_path": '/Users/ChengBoon/Documents/GitHub/FYP_Final/ObjectDetector/models/yolov8n.pt',
		"model_type" : ObjectModelType.YOLOV8,
		"classes_path" : '/Users/ChengBoon/Documents/GitHub/FYP_Final/ObjectDetector/models/coco_label.txt',
		"box_score" : 0.1,
		"box_nms_iou" : 0.1
	}

	YoloDetector.set_defaults(config)
	network = YoloDetector()

	get_colors = list(map(lambda i:"#" +"%06x" % random.randint(0, 0xFFFFFF),range(len(network.class_names)) ))
	colors_dict = dict(zip(list(network.class_names), get_colors))

	fps = 0
	frame_count = 0
	start = time.time()
	while True:
		_, frame = capture.read()
		if cv2.waitKey(1)==27 or frame is None:    # Esc key to stop
			print("End of stream.", logging.INFO)
			break
		
		network.DetectFrame(frame)
		network.DrawDetectedOnFrame(frame)


		frame_count += 1
		if frame_count >= 30:
			end = time.time()
			fps = frame_count / (end - start)
			frame_count = 0
			start = time.time()

		cv2.putText(frame, "FPS: %.2f" % fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow("output", frame)
