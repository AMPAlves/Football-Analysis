import cv2

def read_video(path):
    # Path to the video file
    cap = cv2.VideoCapture(path)
    frames = []
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or cannot read frame.")
                break
            frames.append(frame)
        return frames
    
def get_video_fps(path):
    cap = cv2.VideoCapture(path)
    return cap.get(cv2.CAP_PROP_FPS)
    
def save_video(video_frames,fps,output_path):
    cc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path,cc,fps,(video_frames[0].shape[1],video_frames[0].shape[0]))
    for frame in video_frames:
        output.write(frame)
    output.release()