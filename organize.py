import os
import pprint as pp
import cv2
from random import randrange

path = "C:\Football Analysis\clips"
files = os.listdir(path)

def cut_frames(video,index,frames):
    iterations = (7 if index % 2 == 0 else 8)
    for n in range(iterations):
        selected_frame = randrange(0,frames)
        video.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
        retval, img = video.read()
        file_name =str(index)+"-"+str(n)+".jpeg"
        pp.pprint(file_name)
        cv2.imwrite(os.path.join("to_label_data",file_name),img)
        



for index, file in enumerate(files):
    file_rename = os.path.join(path,str(index)+".mp4")
    if not os.path.isfile(file_rename):
        pp.pprint("Check name change")
        os.rename(os.path.join(path,file), file_rename)
    video = cv2.VideoCapture(file_rename)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    cut_frames(video,index,frame_count)