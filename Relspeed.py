import cv2
import time 
import numpy as np
from playsound import playsound
import http.client, urllib
from pygame import mixer
mixer.init()
sound = mixer.Sound('FYP_Final/assets/alarm.mp3')


def calculate_optical_flow(frame1, frame2):
        """Calculates the optical flow between two frames.

        Args:
            frame1: The first frame.
            frame2: The second frame.

        Returns:
            The optical flow between the two frames.
        """

        # Convert the frames to grayscale.
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Calculate the optical flow.
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            return flow
        except:
            return None

def calculate_relative_speed(flow):
    """Calculates the relative speed of the front vehicle from the optical flow.

    Args:
        flow: The optical flow.

    Returns:
        The relative speed of the front vehicle.
    """

    # Get the x and y components of the optical flow.
    u = flow[..., 0]
    v = flow[..., 1]

    # Calculate the magnitude of the optical flow.
    mag = np.sqrt(u**2 + v**2)

    # Calculate the relative speed of the front vehicle.
    relative_speed = np.mean(mag)

    return relative_speed

def play(): 

    # conn = http.client.HTTPSConnection("api.pushover.net:443")
    if mixer.Sound.get_num_channels(sound) == 0:
        sound.play()

    # conn.request("POST", "/1/messages.json",
    #             urllib.parse.urlencode({
    #                 "token": "aez9em24ipfca4jya1zry5jx13hius",
    #                 "user": "u6beooa6pvkupdr9wmucedvxi97wsm",
    #                 "message": "Warning Collision Alert",
    #             }), { "Content-type": "application/x-www-form-urlencoded" })

        
    
    # conn.getresponse()