import cv2
import numpy as np
import typing  

class SingleCamDistanceMeasure(object):
	RefSizeDict = { 
					"person" : (160, 50), 
					"bicycle" : (98, 65),
					"motorbike" : (100, 100),
					"car" : (150, 180 ),
					"bus" : (319, 250), 
					"truck" : (346, 250), 
				 }

	def __init__(self, object_list=["person", "bicycle", "car", "motorbike", "bus", "truck"] ):
		self.object_list = object_list
		self.focal_length = 13
		self.distance_points = []

	def isInsidePolygon(self, pt, poly ):
		inside = False
		l = len(poly)
		i = -1
		j = l - 1
		while i < l - 1:
			i += 1
			if((poly[i][0]<=pt[0] and pt[0] < poly[j][0]) or (poly[j][0]<=pt[0] and pt[0]<poly[i][0])):
				if(pt[1]<(poly[j][1]-poly[i][1]) * (pt[0]-poly[i][0])/(poly[j][0]-poly[i][0])+poly[i][1]):
					inside = not inside
			j=i
		return inside

	def calcDistance(self, boxes) :
		"""
		Calculate the distance of the target object through the size of pixels.

		Args:
			boxes: coordinate information and labels of the target object.

		Returns:
		"""
		self.distance_points = []
		if (len(boxes) != 0 ):
			for box, _ in boxes:
				ymin, xmin, ymax, xmax, label = box
				if label in self.object_list and xmax - xmin < 1280-50:
					point_x = (xmax + xmin) // 2
					point_y = ymin
					try :
						distance = (self.RefSizeDict[label][1] * self.focal_length)/ (xmax - xmin)
						self.distance_points.append([point_x, point_y, distance])
					except :
						pass
 
	def calcCollisionPoint(self, poly):
		"""
		Determine whether the target object is within the main lane lines.

		Args:
			poly: is a polygonal points. [[x1, y1], [x2, y2], [x3, y3] ... [xn, yn]]

		Returns:
			[Xcenter, Ybottom, distance]
		"""
		if (len(self.distance_points) != 0 and len(poly) )  :
			self.distance_points.sort()
			for x, y, d in self.distance_points:
				if (self.isInsidePolygon( (x, y), np.squeeze(poly))):
					return [x, y, d]
		return None

	def DrawDetectedOnFrame(self, frame_show) :
		if (len(self.distance_points) != 0 )  :
			for x, y, d in self.distance_points:
				cv2.circle(frame_show, (x, y), 4, (0, 255 , 0), thickness=-1)

				unit = 'm'
				if d < 0: text = ' {} {}'.format( "unknown", unit)
				else: text = ' {:.2f} {}'.format(d, unit)
				
				if (d > 3): fontScale = 0.5
				elif (1.5 < d <= 3): fontScale = 0.8
				elif (d <= 1.5): fontScale = 1
				else: fontScale = 0.8
      
				# get coords based on boundary
				textsize = cv2.getTextSize(text, 0, fontScale=fontScale, thickness=3)[0]
				textX = int((x - textsize[0]/2))
				textY = int((y + textsize[1]))

				cv2.putText(frame_show, text, (textX, textY), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=fontScale,  color=(0, 255 , 0), thickness=2)
		return frame_show
