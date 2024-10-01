import supervision as sv
from inference import get_model
import cv2
import numpy
from key_utils import API_ROBOFLOW
import numpy as np

def infer_pitch_keypoints(model_path,video_frames):
    
    output_frames = []
    Pitch_Keypoints_Model = get_model(model_id=model_path,api_key=API_ROBOFLOW)
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex("#008BF8"))
    
    
    for frame in video_frames:
        
        frame_inference = Pitch_Keypoints_Model.infer(frame, confidence=0.5)[0]
        frame_detections = sv.KeyPoints.from_inference(frame_inference)
        confidence_frames = frame_detections.confidence[0] > 0.5
        frame_detections_key_points = frame_detections.xy[0][confidence_frames]
        frame_keypoints = sv.KeyPoints(xy=frame_detections_key_points[np.newaxis, ...])
        
        
        annotated_frame = frame.copy()
        annotated_frame = vertex_annotator.annotate(scene=annotated_frame,key_points=frame_keypoints)
        opencvImage = cv2.cvtColor(numpy.array(annotated_frame), cv2.COLOR_RGB2BGR)
        opencvImage = opencvImage[:, :, ::-1].copy()
        output_frames.append(opencvImage)
    
    
    return output_frames    