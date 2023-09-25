import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from copy import copy
from moviepy.editor import VideoFileClip


class Lane():
    def __init__(self, N):
        # left_lanes (np.array): Left lanes points.[[x1, y1], [x2, y2], [x3, y3] ... [xn, yn]]
        self.lanes = []
        # was the line detected in the last frame or not
        self.detected = False
        # x values for detected line pixels
        self.cur_fitx = None
        # y values for detected line pixels
        self.cur_fity = None
        # x values of the last N fits of the line
        self.prev_fitx = []
        # polynomial coefficients for the most recent fit
        self.current_poly = [np.array([False])]
        # best polynomial coefficients for the last iteration
        self.prev_poly = [np.array([False])]
        self.N = N
        self.draw_area = False

    def average_pre_lanes(self):
        tmp = copy(self.prev_fitx)
        tmp.append(self.cur_fitx)
        self.mean_fitx = np.mean(tmp, axis=0)

    def append_fitx(self):
        if len(self.prev_fitx) == self.N:
            self.prev_fitx.pop(0)
        self.prev_fitx.append(self.mean_fitx)

    def process(self, ploty):
        self.cur_fity = ploty
        self.average_pre_lanes()
        self.append_fitx()
        self.prev_poly = self.current_poly


class LaneDetector():
    def __init__(self, horizon=480, left_x_offset=100, right_x_offset=100, bottom_l=400, bottom_r=400, bottom=0):
        self.frame_width = 1280
        self.frame_height = 960
        self.LANEWIDTH = 3.5  # highway lane width in US: 3.7 meters
        self.input_scale = 2
        self.output_frame_scale = 1
        self.N = 4  # buffer previous N lines

        self.left_lane = Lane(self.N)
        self.right_lane = Lane(self.N)
        self.pts = []
        self.offcenter = 0

        # fullsize:1280x960
        # self.x = [0-400, self.frame_width+400, round((self.frame_width/2)+100), round((self.frame_width/2))-40]
        # self.y = [self.frame_height, self.frame_height, round(self.frame_height/2)-45, round(self.frame_height/2)-45]
        
        self.x = [0+bottom_l, self.frame_width+bottom_r, round((self.frame_width/2)+right_x_offset), round((self.frame_width/2))+left_x_offset]
        self.y = [self.frame_height+bottom, self.frame_height+bottom, horizon, horizon]
        
        self.X = [350, 990, 990, 350]
        self.Y = [self.frame_height, self.frame_height, 0, 0]
        
        x, y = self.x, self.y
        X, Y = self.X, self.Y

        self.src = np.floor(np.float32([
            [x[3], y[3]],    # top-left
            [x[0], y[0]],   # bottom-left
            [x[1], y[1]],   # bottom-right
            [x[2], y[2]]   # top-right
        ]) / self.input_scale)
        self.dst = np.floor(np.float32([
            [X[3], Y[3]],    # top-left
            [X[0], Y[0]],   # bottom-left
            [X[1], Y[1]],   # bottom-right
            [X[2], Y[2]]   # top-right
        ]) / self.input_scale)

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def tracker(self, binary_sub, ploty, input_scale):
        left_fit, right_fit = window_search(self.left_lane.prev_poly, self.right_lane.prev_poly, binary_sub, margin=100/self.input_scale, input_scale=self.input_scale)

        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + \
            right_fit[1] * ploty + right_fit[2]

        std_value = np.std(right_fitx - left_fitx)
        if std_value < (85 / self.input_scale) and std_value != 0:
            self.left_lane.detected = True
            self.right_lane.detected = True
            self.left_lane.current_poly = left_fit
            self.right_lane.current_poly = right_fit
            self.left_lane.cur_fitx = left_fitx
            self.right_lane.cur_fitx = right_fitx
        else:
            self.left_lane.detected = False
            self.right_lane.detected = False
            self.left_lane.current_poly = self.left_lane.prev_poly
            self.right_lane.current_poly = self.right_lane.prev_poly
            self.left_lane.cur_fitx = self.left_lane.prev_fitx[-1]
            self.right_lane.cur_fitx = self.right_lane.prev_fitx[-1]

    def detector(self, binary_sub, ploty, visualization=False):
        left_fit, right_fit = full_search(binary_sub, input_scale=self.input_scale)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + \
            right_fit[1] * ploty + right_fit[2]
        std_value = np.std(right_fitx - left_fitx)
        if std_value < (85 / self.input_scale) and std_value != 0:
            self.left_lane.current_poly = left_fit
            self.right_lane.current_poly = right_fit
            self.left_lane.cur_fitx = left_fitx
            self.right_lane.cur_fitx = right_fitx
            self.left_lane.detected = True
            self.right_lane.detected = True
        else:
            self.left_lane.current_poly = self.left_lane.prev_poly
            self.right_lane.current_poly = self.right_lane.prev_poly
            if len(self.left_lane.prev_fitx) > 0:
                self.left_lane.cur_fitx = self.left_lane.prev_fitx[-1]
                self.right_lane.cur_fitx = self.right_lane.prev_fitx[-1]
            else:
                self.left_lane.cur_fitx = left_fitx
                self.right_lane.cur_fitx = right_fitx
            self.left_lane.detected = False
            self.right_lane.detected = False

    def DrawAreaOnFrame(self, image, offcenter=0, color=(255, 191, 0)):
        self.draw_area = True
        # Draw a mask for the current lane
        if abs(offcenter) > 0.6: color = (0, 0, 255) # red
        else: color = (255, 191, 0) # blue
        color_warp = np.zeros_like(image).astype(np.uint8)
        cv2.fillPoly(color_warp, np.int_([self.pts]), color)  
        newwarp = cv2.warpPerspective(color_warp, self.M_inv, (int(self.frame_width/self.input_scale), int(self.frame_height/self.input_scale)))
        newwarp_ = cv2.resize(newwarp, None, fx=self.input_scale/self.output_frame_scale, fy=self.input_scale/self.output_frame_scale, interpolation=cv2.INTER_LINEAR)

        image = cv2.addWeighted(image, 1, newwarp_, 0.2, 0)
        if (not self.draw_area): self.pts = []
        return image

    def process_frame(self, img, visualization=False):
        img_undist = cv2.resize( img, (0, 0), fx=1/self.input_scale, fy=1/self.input_scale)

        # find the binary image of lane/edges
        img_binary = find_edges(img_undist)

        # warp the image to bird view
        # get binary image contains edges
        binary_warped = warper(img_binary, self.M)

        # crop the binary image
        binary_sub = np.zeros_like(binary_warped)
        binary_sub[:, int(150/self.input_scale):int(-80/self.input_scale)] = binary_warped[:, int(150/self.input_scale):int(-80/self.input_scale)]

        # start detector or tracker to find the lanes
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fit, right_fit = None, None
        if self.left_lane.detected or self.right_lane.detected: self.tracker(binary_sub, ploty, visualization) # start tracker
        else: self.detector(binary_sub, ploty, visualization) # start detector
        
        # average among the previous N frames to get the averaged lanes
        self.left_lane.process(ploty)
        self.right_lane.process(ploty)

        pts_left = np.array([np.transpose(np.vstack([self.left_lane.mean_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_lane.mean_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        self.pts = pts

        return pts, pts_left, pts_right


# Threshold for color and gradient thresholding
s_thresh, sx_thresh, dir_thresh, r_thresh = (120, 255), (20, 100), (0.7, 1.3), (200, 255)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute( cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute( cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def threshold_col_channel(channel, thresh):
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary


import cv2
import numpy as np
from scipy import signal


def abs_sobel(img_ch, orient='x', sobel_kernel=3):
    """
    Applies the sobel operation on a gray scale image.

    :param img_ch:
    :param orient: 'x' or 'y'
    :param sobel_kernel: an uneven integer
    :return:
    """
    if orient == 'x':
        axis = (1, 0)
    elif orient == 'y':
        axis = (0, 1)
    else:
        raise ValueError('orient has to be "x" or "y" not "%s"' % orient)

    sobel = cv2.Sobel(img_ch, -1, *axis, ksize=sobel_kernel)
    abs_s = np.absolute(sobel)

    return abs_s


def gradient_magnitude(sobel_x, sobel_y):
    """
    Calculates the magnitude of the gradient.
    :param sobel_x:
    :param sobel_y:
    :return:
    """
    abs_grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return abs_grad_mag.astype(np.uint16)


def gradient_direction(sobel_x, sobel_y):
    """
    Calculates the direction of the gradient. NaN values cause by zero division will be replaced
    by the maximum value (np.pi / 2).
    :param sobel_x:
    :param sobel_y:
    :return:
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))
        abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2

    return abs_grad_dir.astype(np.float32)


def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel
    :param img:
    :param kernel_size:
    :return:
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def extract_yellow(img):
    """
    Generates an image mask selecting yellow pixels.
    :param img: image with pixels in range 0-255
    :return: Yellow 255 not yellow 0
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))

    return mask


def extract_dark(img):
    """
    Generates an image mask selecting dark pixels.
    :param img: image with pixels in range 0-255
    :return: Dark 255 not dark 0
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0.), (255, 153, 128))
    return mask


def extract_highlights(img, p=99.9):
    """
    Generates an image mask selecting highlights.
    :param p: percentile for highlight selection. default=99.9
    :param img: image with pixels in range 0-255
    :return: Highlight 255 not highlight 0
    """
    p = int(np.percentile(img, p) - 30)
    mask = cv2.inRange(img, p, 255)
    return mask


def binary_noise_reduction(img, thresh):
    """
    Reduces noise of a binary image by applying a filter which counts neighbours with a value
    and only keeping those which are above the threshold.
    :param img: binary image (0 or 1)
    :param thresh: min number of neighbours with value
    :return:
    """
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(img, ddepth=-1, kernel=k)
    img[nb_neighbours < thresh] = 0
    return img


def generate_lane_mask(img, v_cutoff=0):
    """
    Generates a binary mask selecting the lane lines of an street scene image.
    :param img: RGB color image
    :param v_cutoff: vertical cutoff to limit the search area
    :return: binary mask
    """
    window = img[v_cutoff:, :, :]
    yuv = cv2.cvtColor(window, cv2.COLOR_RGB2YUV)
    yuv = 255 - yuv
    hls = cv2.cvtColor(window, cv2.COLOR_RGB2HLS)
    chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)
    gray = np.mean(chs, 2)

    s_x = abs_sobel(gray, orient='x', sobel_kernel=3)
    s_y = abs_sobel(gray, orient='y', sobel_kernel=3)

    grad_dir = gradient_direction(s_x, s_y)
    grad_mag = gradient_magnitude(s_x, s_y)

    ylw = extract_yellow(window)
    highlights = extract_highlights(window[:, :, 0])

    mask = np.zeros(img.shape[:-1], dtype=np.uint8)

    mask[v_cutoff:, :][((s_x >= 25) & (s_x <= 255) &
                        (s_y >= 25) & (s_y <= 255)) |
                       ((grad_mag >= 30) & (grad_mag <= 512) &
                        (grad_dir >= 0.2) & (grad_dir <= 1.)) |
                       (ylw == 255) |
                       (highlights == 255)] = 1

    mask = binary_noise_reduction(mask, 4)

    return mask

def find_edges(img, s_thresh=s_thresh, sx_thresh=sx_thresh, dir_thresh=dir_thresh):
    img = np.copy(img)
    # Convert to HSL color space and threshold the s channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    s_channel = hls[:, :, 2]
    s_binary = threshold_col_channel(s_channel, thresh=s_thresh)

    # Sobel x
    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)
    # # gradient direction
    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)
    # # output mask
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(((sxbinary == 1) & (dir_binary == 1)) |
                     ((s_binary == 1) & (dir_binary == 1)))] = 1
    # add more weights for the s channel
    c_bi = np.zeros_like(s_channel)
    c_bi[((sxbinary == 1) & (s_binary == 1))] = 2

    ave_binary = (combined_binary + c_bi)
    return ave_binary


def warper(img, M):
    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped


def full_search(binary_warped, input_scale):
    # histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] // 2):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = out_img.astype('uint8')

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int32(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int32(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = np.floor(100/input_scale)
    # Set minimum number of pixels found to recenter window
    minpix = np.floor(50/input_scale)
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        left_fit = np.array([1, 1, 1])
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        right_fit = np.array([1, 1, 1])

    return left_fit, right_fit


def window_search(left_fit, right_fit, binary_warped, input_scale, margin=100):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's easier to find line pixels with windows search
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        left_fit = np.array([1, 1, 1])
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        right_fit = np.array([1, 1, 1])

    return left_fit, right_fit


def measure_lane_curvature(ploty, leftx, rightx, frame_height, input_scale):

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    # meters per pixel in y dimension
    ym_per_pix = 30/(frame_height/input_scale)
    xm_per_pix = 3.5/(frame_width/input_scale)  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                     left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                      right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    if leftx[0] - leftx[-1] > 50/input_scale:
        curve_direction = 'Left curve'
    elif leftx[-1] - leftx[0] > 50/input_scale:
        curve_direction = 'Right curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad+right_curverad)/2.0, curve_direction


def off_center(left, mid, right):
    LANEWIDTH = 3.5

    a = mid - left
    b = right - mid
    width = right - left

    if a >= b:  # driving right off
        offset = a / width * LANEWIDTH - LANEWIDTH / 2.0
    else:       # driving left off
        offset = LANEWIDTH / 2.0 - b / width * LANEWIDTH

    return offset


def compute_car_offcenter(ploty, left_fitx, right_fitx, undist):

    # Create an image to draw the lines on
    height = undist.shape[0]
    width = undist.shape[1]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    bottom_l = left_fitx.max()
    bottom_r = right_fitx.max()

    offcenter = off_center(bottom_l, width/2.0, bottom_r)

    return offcenter, pts, pts_left, pts_right


def draw_lane(frame, offcenter, pts, curvature, curve_direction, input_scale, output_frame_scale, font=cv2.FONT_HERSHEY_SIMPLEX):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    x = [0-400, frame_width+400,
         round((frame_width/2)+150), round((frame_width/2)-50)]
    y = [frame_height, frame_height, round(
        frame_height/2)-40, round(frame_height/2)-40]
    X = [290, 990, 990, 290]
    Y = [frame_height, frame_height, 0, 0]

    src = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]], [
                   x[2], y[2]], [x[3], y[3]]]) / input_scale)
    dst = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]], [
                   X[2], Y[2]], [X[3], Y[3]]]) / input_scale)

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    if offcenter >= 0:
        offset = offcenter
        direction = 'Right'
    elif offcenter < 0:
        offset = -offcenter
        direction = 'Left'

    info_road = "Road Status"
    info_lane = "Lane info: {0}".format(curve_direction)
    info_cur = "Curvature {0:6.1f} m".format(curvature)
    info_offset = "Off center: {0} {1:3.2f}m".format(direction, offset)

    cv2.putText(frame, "Departure Warning System with a Monocular Camera",
                (23, 25), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, info_lane, (25, 80+40), font,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, info_cur, (25, 80+80), font,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, info_offset, (25, 80+120),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    color_warp = np.zeros_like(frame).astype(np.uint8)
    if abs(offcenter) > 0.6:  # car is offcenter more than 0.6 m
        # Draw Red lane
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))  # red
    else:  # Draw Green lane
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))  # green

    newwarp = cv2.warpPerspective(color_warp, M_inv, (int(
        frame_width/input_scale), int(frame_height/input_scale)))
    bird = cv2.line(newwarp, (X[0], Y[0]), (X[1], Y[1]), (0, 0, 255))
    bird = cv2.line(bird, (X[1], Y[1]), (X[2], Y[2]), (0, 0, 255))
    bird = cv2.line(bird, (X[2], Y[2]), (X[3], Y[3]), (0, 0, 255))
    bird = cv2.line(bird, (X[3], Y[3]), (X[0], Y[0]), (0, 0, 255))
    newwarp_ = cv2.resize(newwarp, None, fx=input_scale/output_frame_scale,
                          fy=input_scale/output_frame_scale, interpolation=cv2.INTER_LINEAR)

    frame = cv2.addWeighted(frame, 1, newwarp_, 0.3, 0)
    cv2.imshow("Bird", bird)
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture('test_right_2.mov')
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    horizon, top_l, top_r, bottom_l, bottom_r, bottom = 435, -80, 150, -300, 400, -80
    laneDetector = LaneDetector(horizon=horizon, left_x_offset=top_l, right_x_offset=top_r, bottom_l=bottom_l, bottom_r=bottom_r, bottom=bottom)
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_show = frame.copy()
        width, height = (1280, 960)
        if ret:
            frame_show = cv2.resize(frame_show, (width, height))

            pts, pts_left, pts_right = laneDetector.process_frame(frame_show, False)

            frame_show = laneDetector.DrawAreaOnFrame(frame_show)
            cv2.imshow('frame', frame_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break