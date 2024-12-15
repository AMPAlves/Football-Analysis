from ultralytics import YOLO
import pickle
import os
import supervision as sv

class Tracker:
    def __init__(self, model):
        self.model = YOLO(model)
        self.tracker = sv.ByteTrack()
        
    def infer_frames(self, frames):
        batch = 24
        detection = []
        for i in range(0,len(frames),batch):
            detection_batch = self.model(frames[i:i+batch])
            detection += detection_batch
        return detection
        
    
    def get_trackingboxes(self, frames, tracks_path=None):
        
        if not tracks_path == None and os.path.exists(tracks_path):
            with open(tracks_path,"rb") as file:
                tracks = pickle.load(file)
            return tracks
         
        tracks = {
            "players": [],
            "goalkeepers": [],
            "referees": [],
            "ball": []
        }
        
        detections = self.infer_frames(frames)
        for index, detection in enumerate(detections):
            labels = {v:k for k,v in detection.names.items()}
            detection_sv = sv.Detections.from_transformers(detection)
            
            detection_tracker = self.tracker.update_with_detections(detection_sv)
            tracks["players"].append({})
            tracks["goalkeepers"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_tracker:
                box = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if class_id == labels["Player"]:
                    tracks["players"][index][track_id]={"bbox":box}
                if class_id == labels["Referee"]:
                    tracks["referees"][index][track_id]={"bbox":box}
                if class_id == labels["Goalkeeper"]:
                    tracks["goalkeepers"][index][track_id]={"bbox":box}
                if class_id == labels["Football Ball"]:
                    tracks["ball"][index][1]={"bbox":box}

        if not tracks_path == None and not os.path.exists(tracks_path):
            with open(tracks_path, "wb") as file:
                pickle.dump(tracks,file)
                
        return tracks
    
    def draw_bounding_boxes(self,frames,tracks):
        output_frames = []
        box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['##FAB3A9', '#2B59C3', 'A4243B', 'FF934F']))

        for index, frame in enumerate(frames):
            frame = frame.copy()
            frame = box_annotator.annotate(scene=frame,detections=tracks)
            sv.plot_image(frame)
            break        
                
            
            
            