import supervision as sv
from ultralytics import YOLO
import cv2
import numpy
from PIL import Image

def frame_generator(model_path
                    ,video_frames):
    
    output_frames = []
    model = YOLO(model_path)
    
    ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#2B59C3', '#A4243B', '#FF934F']),start_angle=45)
    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['#FAB3A9']),thickness=4)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#FAB3A9', '#2B59C3', '#A4243B', '#FF934F']), text_color=sv.Color.from_hex('#000000'))
    
    for frame in video_frames:
        frame_inference = model.predict(frame,conf=0.25)[0]
        frame_detections = sv.Detections.from_ultralytics(frame_inference)
        
        ball_detections = frame_detections[frame_detections.class_id == 0]
        noBall_detections = frame_detections[frame_detections.class_id != 0]
        noBall_detections = noBall_detections.with_nms(class_agnostic=True)
        noBall_detections.class_id -= 1
        
        labels= [
            f'{class_name} | {class_id} | {confidence:.2f}'
            for class_name, class_id, confidence in zip(frame_detections['class_name'], frame_detections.class_id, frame_detections.confidence)
        ]
        
        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame,detections=noBall_detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame,detections=ball_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame,detections=frame_detections,labels=labels)
        opencvImage = cv2.cvtColor(numpy.array(annotated_frame), cv2.COLOR_RGB2BGR)
        opencvImage = opencvImage[:, :, ::-1].copy()
        output_frames.append(opencvImage)
    
    return output_frames
    
def frame_generator_with_tracker(model_path
                    ,video_frames):
    
    output_frames = []
    model = YOLO(model_path)
    
    ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#2B59C3', '#A4243B', '#FF934F']),start_angle=45)
    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['#FAB3A9']),thickness=4)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#2B59C3', '#A4243B', '#FF934F']), text_color=sv.Color.from_hex('#000000'))
    
    byteTracker = sv.ByteTrack(track_activation_threshold=0.3, frame_rate=25)
    
    for frame in video_frames:
        frame_inference = model.predict(frame,conf=0.25)[0]
        frame_detections = sv.Detections.from_ultralytics(frame_inference)
        
        ball_detections = frame_detections[frame_detections.class_id == 0]
        
        noBall_detections = frame_detections[frame_detections.class_id != 0]
        noBall_detections = noBall_detections.with_nms(class_agnostic=True)
        noBall_detections.class_id -= 1
        noBall_detections = byteTracker.update_with_detections(detections=noBall_detections)
        
        tracker_labels = [
            f'{tracker_id}'
            for tracker_id in noBall_detections.tracker_id
        ]
        
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame,detections=ball_detections)
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame,detections=noBall_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame,detections=noBall_detections,labels=tracker_labels)
        
        opencvImage = cv2.cvtColor(numpy.array(annotated_frame), cv2.COLOR_RGB2BGR)
        opencvImage = opencvImage[:, :, ::-1].copy()
        output_frames.append(opencvImage)
        
        
    
    return output_frames
    