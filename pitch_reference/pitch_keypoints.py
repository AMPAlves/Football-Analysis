import supervision as sv
from inference import get_model
import cv2
import numpy
from key_utils import API_ROBOFLOW
import numpy as np
from pitch_dimension import SoccerPitch, PitchTransformation

def infer_pitch_keypoints(model_path,video_frames):
    
    VertexInfo = SoccerPitch()
    output_frames = []
    Pitch_Keypoints_Model = get_model(model_id=model_path,api_key=API_ROBOFLOW)
    vertex_annotator = sv.VertexLabelAnnotator(color=sv.Color.from_hex("#008BF8"), text_color=sv.Color.from_hex('#000000'))
    #label_annotator = sv.LabelAnnotator(color=sv.Color.from_hex("#008BF8"), text_color=sv.Color.from_hex('#000000'))
    
    #byteTracker = sv.ByteTrack(track_activation_threshold=0.3, frame_rate=25)
    
    for frame in video_frames:
        
        labels = []
        frame_inference = Pitch_Keypoints_Model.infer(frame, confidence=0.5)[0]
        frame_detections = sv.KeyPoints.from_inference(frame_inference)
        confidence_frames = frame_detections.confidence[0] > 0.5
        frame_detections_key_points = frame_detections.xy[0][confidence_frames]
        frame_keypoints = sv.KeyPoints(xy=frame_detections_key_points[np.newaxis, ...])
        pitch_keypoints_labeled = np.array(VertexInfo.vertices)[confidence_frames]
        
        coordinates = PitchTransformation(source=pitch_keypoints_labeled,target=frame_detections_key_points)
        
        pitch_all_points = np.array(VertexInfo.vertices)
        frame_all_points = coordinates.dimension_coordinates_transformation(points=pitch_all_points)
        
        annotated_frame = frame.copy()
        annotated_frame = vertex_annotator.annotate(scene=annotated_frame,key_points=frame_keypoints,labels=labels)
        opencvImage = cv2.cvtColor(numpy.array(annotated_frame), cv2.COLOR_RGB2BGR)
        opencvImage = opencvImage[:, :, ::-1].copy()
        #cv2.imwrite("C:/Football Analysis/corno.png", opencvImage)
        output_frames.append(opencvImage)
    
    return output_frames

def infer_pitch_model_to_supervision(model_path,video_frames):
    
    VertexInfo = SoccerPitch()
    output_frames = []
    Pitch_Keypoints_Model = get_model(model_id=model_path,api_key=API_ROBOFLOW)
    vertex_annotator = sv.VertexLabelAnnotator(color=sv.Color.from_hex("#008BF8"), text_color=sv.Color.from_hex('#000000'))
    #label_annotator = sv.LabelAnnotator(color=sv.Color.from_hex("#008BF8"), text_color=sv.Color.from_hex('#000000'))
    
    #byteTracker = sv.ByteTrack(track_activation_threshold=0.3, frame_rate=25)
    
    for frame in video_frames:
        
        pitch_frame_inference = Pitch_Keypoints_Model.infer(frame, confidence=0.5)[0]
        pitch_frame_detections = sv.KeyPoints.from_inference(pitch_frame_inference)
        confidence_frames = pitch_frame_detections.confidence[0] > 0.5
        frame_detections_key_points = pitch_frame_detections.xy[0][confidence_frames]
        frame_keypoints = sv.KeyPoints(xy=frame_detections_key_points[np.newaxis, ...])
        pitch_keypoints_labeled = np.array(VertexInfo.vertices)[confidence_frames]
        
        coordinates = PitchTransformation(source=pitch_keypoints_labeled,target=frame_detections_key_points)