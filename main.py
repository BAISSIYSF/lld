"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
#from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
import matplotlib.pyplot as plt
np.int = int
class FindLaneLines:
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        
        # print(img.shape)
        # print(img.dtype)
        img = cv2.resize(img, (1280, 720))
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        out_img = np.copy(img)
        # cv2.imwrite('frm1.png', img)
        # exit()
        
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        # plt.pause(0)
        # plt.clf() 
        img = self.calibration.undistort(img)
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        # plt.pause(0)
        # plt.clf() 
        img = self.transform.forward(img)
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        # plt.pause(0)
        # plt.clf() 
        img = self.thresholding.forward(img)
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        # plt.pause(0)
        # plt.clf() 
        img = self.lanelines.forward(img)
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        # plt.pause(0)
        # plt.clf() 
        img = self.transform.backward(img)
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        # plt.pause(0)
        # plt.clf() 

        if out_img.dtype == 'float32':
            out_img = (out_img * 255).astype(np.uint8)
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        # plt.imshow(out_img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        # plt.pause(0)
        # plt.clf() 
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

def main():
    findLaneLines = FindLaneLines()
    findLaneLines.process_video('test2.mp4', 'output.mp4')


if __name__ == "__main__":
    main()