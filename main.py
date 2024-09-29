from video.video_utils import read_video, get_video_fps, save_video
from supervision_utils import frame_generator

def main():
    
    video_path = 'originaldata/clips/20.mp4'
    model_path = 'models/best.pt'
    video_frames = read_video(video_path)
    video_fps = get_video_fps(video_path)

    annotated_frames = frame_generator(model_path=model_path,video_frames=video_frames)
    
    save_video(annotated_frames,video_fps,output_path='output_prediction/output.mp4')
    
    
if __name__ == '__main__':
    main()