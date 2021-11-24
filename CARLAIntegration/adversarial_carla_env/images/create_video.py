import numpy as np
import glob
import cv2
import os

# settings
frame_rate = 20

dirname = os.path.abspath(os.path.dirname(__file__))
print(dirname)
image_folder = os.path.join(dirname, 'ego')
video_name_1 = 'carla_video.avi'

# get number of frames
png_counter = int(len(glob.glob1(image_folder,"*.png"))/2)

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

#clean_image
video = cv2.VideoWriter(video_name_1, 0, frame_rate, (width*2,height*3)) # notice width/height multipliers
for i in range(png_counter):
    s0 = os.path.join(dirname, "spectator",  str(i) + "_spectator.png")
    s1 = os.path.join(dirname, "ego", str(i) + "_before_image.png")
    s2 = os.path.join(dirname, "ego", str(i) + "_after_image.png")
    img0 = cv2.imread(s0)
    img1 = cv2.imread(s1)
    img2 = cv2.imread(s2)
    if img0 is not None and img1 is not None and img2 is not None:
        side_by_side_img = np.concatenate((img1,img2),axis=1)
        complete_img = np.concatenate((side_by_side_img,img0),axis=0)
        video.write(complete_img)

cv2.destroyAllWindows()
video.release()
