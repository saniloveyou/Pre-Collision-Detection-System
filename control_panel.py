
import os, cv2, time
import numpy as np
import logging
from ObjectDetector.utils import CollisionType
from LaneDetector.utils import OffsetType, CurvatureType
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.mp3')

class ControlPanel(object):
	CollisionDict = {
						CollisionType.UNKNOWN : (0, 255, 255),
						CollisionType.NORMAL : (0, 255, 0),
						CollisionType.PROMPT : (0, 102, 255),
						CollisionType.WARNING : (0, 0, 255)
	 				}

	OffsetDict = { 
					OffsetType.UNKNOWN : (0, 255, 255), 
					OffsetType.RIGHT :  (0, 0, 255), 
					OffsetType.LEFT : (0, 0, 255), 
					OffsetType.CENTER : (0, 255, 0)
				 }

	CurvatureDict = { 
						CurvatureType.UNKNOWN : (0, 255, 255),
						CurvatureType.STRAIGHT : (0, 255, 0),
						CurvatureType.EASY_LEFT : (0, 102, 255),
						CurvatureType.EASY_RIGHT : (0, 102, 255),
						CurvatureType.HARD_LEFT : (0, 0, 255),
						CurvatureType.HARD_RIGHT : (0, 0, 255)
					}

	def __init__(self):
		collision_warning_img = cv2.imread('./assets/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
		self.collision_warning_img = cv2.resize(collision_warning_img, (100, 100))
		collision_prompt_img = cv2.imread('./assets/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
		self.collision_prompt_img = cv2.resize(collision_prompt_img, (100, 100))
		collision_normal_img = cv2.imread('./assets/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
		self.collision_normal_img = cv2.resize(collision_normal_img, (100, 100))
		left_curve_img = cv2.imread('./assets/left_turn.png', cv2.IMREAD_UNCHANGED)
		self.left_curve_img = cv2.resize(left_curve_img, (200, 200))
		right_curve_img = cv2.imread('./assets/right_turn.png', cv2.IMREAD_UNCHANGED)
		self.right_curve_img = cv2.resize(right_curve_img, (200, 200))
		keep_straight_img = cv2.imread('./assets/straight.png', cv2.IMREAD_UNCHANGED)
		self.keep_straight_img = cv2.resize(keep_straight_img, (200, 200))
		determined_img = cv2.imread('./assets/warn.png', cv2.IMREAD_UNCHANGED)
		self.determined_img = cv2.resize(determined_img, (200, 200))
		left_lanes_img = cv2.imread('./assets/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
		self.left_lanes_img = cv2.resize(left_lanes_img, (300, 200))
		right_lanes_img = cv2.imread('./assets/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)
		self.right_lanes_img = cv2.resize(right_lanes_img, (300, 200))


		# FPS
		self.fps = 0
		self.frame_count = 0
		self.start = time.time()

		self.curve_status = None

	def _updateFPS(self) :
		"""
		Update FPS.

		Args:
			None

		Returns:
			None
		"""
		self.frame_count += 1
		if self.frame_count >= 30:
			self.end = time.time()
			self.fps = self.frame_count / (self.end - self.start)
			self.frame_count = 0
			self.start = time.time()

	def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25) :
		"""
		Display BirdView Panel on image.

		Args:
			main_show: video image.
			min_show: bird view image.
			show_ratio: display scale of bird view image.

		Returns:
			main_show: Draw bird view on frame.
		"""
		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		min_birdview_show = cv2.resize(min_show, (W, H))
		min_birdview_show = cv2.copyMakeBorder(min_birdview_show, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # 添加边框
		main_show[0:min_birdview_show.shape[0], -min_birdview_show.shape[1]: ] = min_birdview_show
		return main_show
	
	def DisplaySignsPanel(self, main_show, offset_type, curvature_type) :
		"""
		Display Signs Panel on image.

		Args:
			main_show: image.
			offset_type: offset status by OffsetType. (UNKNOWN/CENTER/RIGHT/LEFT)
			curvature_type: curature status by CurvatureType. (UNKNOWN/STRAIGHT/HARD_LEFT/EASY_LEFT/HARD_RIGHT/EASY_RIGHT)

		Returns:
			main_show: Draw sings info on frame.
		"""

		W = 400
		H = 365
		widget = np.copy(main_show[:H, :W])
		widget //= 2
		widget[0:3,:] = [0, 0, 255]  # top
		widget[-3:-1,:] = [0, 0, 255] # bottom
		widget[:,0:3] = [0, 0, 255]  #left
		widget[:,-3:-1] = [0, 0, 255] # right
		main_show[:H, :W] = widget

		if curvature_type == CurvatureType.UNKNOWN and offset_type in { OffsetType.UNKNOWN, OffsetType.CENTER } :
			y, x = self.determined_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.determined_img[y, x, :3]
			self.curve_status = None

		elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status== "Left") and \
			(curvature_type not in { CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT }) :
			y, x = self.left_curve_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.left_curve_img[y, x, :3]
			self.curve_status = "Left"

		elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status== "Right") and \
			(curvature_type not in { CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT }) :
			y, x = self.right_curve_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.right_curve_img[y, x, :3]
			self.curve_status = "Right"
		
		
		if ( offset_type == OffsetType.RIGHT ) :
			y, x = self.left_lanes_img[:,:,2].nonzero()
			main_show[y+10, x-150+W//2] = self.left_lanes_img[y, x, :3]
		elif ( offset_type == OffsetType.LEFT ) :
			y, x = self.right_lanes_img[:,:,2].nonzero()
			main_show[y+10, x-150+W//2] = self.right_lanes_img[y, x, :3]
		elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight" :
			y, x = self.keep_straight_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.keep_straight_img[y, x, :3]
			self.curve_status = "Straight"

		# info_offset = "Off center: {0} {1:3.2f}m".format(direction, offset)
  
		self._updateFPS()
		cv2.putText(main_show, "LDWS : " + offset_type.value, (10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.OffsetDict[offset_type], thickness=2)
		cv2.putText(main_show, "LKAS : " + curvature_type.value, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.CurvatureDict[curvature_type], thickness=2)
		cv2.putText(main_show, "FCWS > LDWS > LKAS", org=(10, 320), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255,255,255), thickness=2)
		cv2.putText(main_show, "FPS  : %.2f" % self.fps, (10, widget.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
		return main_show

	def DisplayCollisionPanel(self, main_show, collision_type, obect_infer_time, lane_infer_time, show_ratio=0.25) :
		"""
		Display Collision Panel on image.

		Args:
			main_show: image.
			collision_type: collision status by CollisionType. (WARNING/PROMPT/NORMAL)
			obect_infer_time: object detection time -> float.
			lane_infer_time:  lane detection time -> float.

		Returns:
			main_show: Draw collision info on frame.
		"""

		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		widget = np.copy(main_show[H+20:2*H, -W-20:])
		widget //= 2
		widget[0:3,:] = [0, 0, 255]  # top
		widget[-3:-1,:] = [0, 0, 255] # bottom
		widget[:,-3:-1] = [0, 0, 255] #left
		widget[:,0:3] = [0, 0, 255]  # right
		main_show[H+20:2*H, -W-20:] = widget

		if (collision_type == CollisionType.WARNING):
			if mixer.Sound.get_num_channels(sound) == 0: mixer.Sound.play(sound, maxtime=580)
			y, x = self.collision_warning_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_warning_img[y, x, :3]
		elif (collision_type == CollisionType.PROMPT) :
			y, x =self.collision_prompt_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_prompt_img[y, x, :3]
		elif (collision_type == CollisionType.NORMAL) :
			y, x = self.collision_normal_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_normal_img[y, x, :3]

		cv2.putText(main_show, "FCWS : " + collision_type.value, ( main_show.shape[1]- int(W) + 100 , 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=self.CollisionDict[collision_type], thickness=2)
		cv2.putText(main_show, "object-infer : %.2f s" % obect_infer_time, ( main_show.shape[1]- int(W) + 100, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
		cv2.putText(main_show, "lane-infer : %.2f s" % lane_infer_time, ( main_show.shape[1]- int(W) + 100, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
		return main_show

