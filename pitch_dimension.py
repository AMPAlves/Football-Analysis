import cv2
import numpy as np

class SoccerPitch:
    pitch_height = 10500
    pitch_width = 6800
    pitch_goal = 732
    pitch_penalty_mark = 1100
    pitch_penalty_area_half = 1650
    pitch_penalty_area_full = 4032
    pitch_goal_area = 600
    pitch_center_radius = 915
    pitch_penalty_arc_from_center = 365
    pitch_corner_radius = 200
    #(...) Continuar os outros
    
    def __init__(self):
        
        self.vertices = [(0,0), #A (Start Left Penalty Area)
                         ((self.pitch_width-self.pitch_penalty_area_full)/2,0), #B
                         ((self.pitch_width-self.pitch_penalty_area_full)/2 + (self.pitch_penalty_area_half-self.pitch_goal_area),0), #C (Importante)
                         ((self.pitch_width-self.pitch_penalty_area_full)/2 + self.pitch_penalty_area_half+self.pitch_goal+self.pitch_goal_area,0), #D (Importante)
                         (self.pitch_width-(self.pitch_width-self.pitch_penalty_area_full)/2,0), #E 
                         (self.pitch_width,0), #F
                         ((self.pitch_width-self.pitch_penalty_area_full)/2 + (self.pitch_penalty_area_half-self.pitch_goal_area),self.pitch_goal_area), #G (IMPORTANTE)
                         ((self.pitch_width-self.pitch_penalty_area_full)/2 + self.pitch_penalty_area_half+self.pitch_goal+self.pitch_goal_area,self.pitch_goal_area), #H (IMPORTANTE)
                         (self.pitch_width/2,self.pitch_penalty_mark), #I (Penalty Mark)
                         ((self.pitch_width-self.pitch_penalty_area_full)/2,self.pitch_penalty_area_half), #J
                         ((self.pitch_width/2)-self.pitch_penalty_arc_from_center,self.pitch_penalty_area_half), #K
                         ((self.pitch_width/2)+self.pitch_penalty_arc_from_center,self.pitch_penalty_area_half), #L
                         (self.pitch_width-(self.pitch_width-self.pitch_penalty_area_full)/2,self.pitch_penalty_area_half), #M (End Left Penalty Area)
                         (0,self.pitch_height/2), #N (Start Center Line)
                         ((self.pitch_width/2)-self.pitch_center_radius,self.pitch_height/2), #O
                         ((self.pitch_width/2)+self.pitch_center_radius,self.pitch_height/2), #P
                         (self.pitch_width,self.pitch_height/2), #Q (End Center Line)
                         ((self.pitch_width-self.pitch_penalty_area_full)/2,self.pitch_height-self.pitch_penalty_area_half), #R (Start Right Penalty Area)
                         ((self.pitch_width/2)-self.pitch_penalty_arc_from_center,self.pitch_height-self.pitch_penalty_area_half), #S
                         ((self.pitch_width/2)+self.pitch_penalty_arc_from_center,self.pitch_height-self.pitch_penalty_area_half), #T
                         (self.pitch_width-(self.pitch_width-self.pitch_penalty_area_full)/2,self.pitch_height-self.pitch_penalty_area_half), #U
                         (self.pitch_width/2,self.pitch_height-self.pitch_penalty_mark), #V (Penalty Mark)
                         ((self.pitch_width-self.pitch_penalty_area_full)/2 + (self.pitch_penalty_area_half-self.pitch_goal_area),self.pitch_height-self.pitch_goal_area), #W (Este)
                         ((self.pitch_width-self.pitch_penalty_area_full)/2 + (self.pitch_penalty_area_half+self.pitch_goal_area+self.pitch_goal),self.pitch_height-self.pitch_goal_area), #X (Este)
                         (0,self.pitch_height), #Y
                         ((self.pitch_width-self.pitch_penalty_area_full)/2,self.pitch_height), #Z
                         ((self.pitch_width-self.pitch_penalty_area_full)/2 + (self.pitch_penalty_area_half-self.pitch_goal_area),self.pitch_height), #AA
                         ((self.pitch_width-self.pitch_penalty_area_full)/2 + (self.pitch_penalty_area_half+self.pitch_goal_area+self.pitch_goal),self.pitch_height), #BB
                         (self.pitch_width-(self.pitch_width-self.pitch_penalty_area_full)/2,self.pitch_height), #CC
                         (self.pitch_width,self.pitch_height), #DD (End Right Penalty Area)
                         (self.pitch_width/2,(self.pitch_height/2)-self.pitch_center_radius), #EE (Start Center Poles)
                         (self.pitch_width/2,(self.pitch_height/2)+self.pitch_center_radius) #FF (End Center Poles)
                         ]
        self.label = ["A",
                      "B",
                      "C",
                      "D",
                      "E",
                      "F",
                      "G",
                      "H",
                      "I",
                      "J",
                      "K",
                      "L",
                      "M",
                      "N",
                      "O",
                      "P",
                      "Q",
                      "R",
                      "S",
                      "T",
                      "U",
                      "V",
                      "W",
                      "X",
                      "Y",
                      "Z",
                      "AA",
                      "BB",
                      "CC",
                      "DD",
                      "EE",
                      "FF"]
        self.linepoints = [(0,1),(0,13),
                           (1,2),(1,9),
                           (2,3),(2,6),
                           (3,4),(3,7),
                           (4,5),(4,12),
                           (5,16),
                           (9,10),
                           (11,12),
                           (6,7),
                           (10,11),
                           (13,14),(13,24),
                           (15,16),
                           (16,29),
                           (17,18),
                           (18,19),
                           (19,20),
                           (22,23),
                           (24,25),
                           (25,26),
                           (25,17),
                           (26,27),
                           (26,22),
                           (27,28),
                           (27,23),
                           (28,29),
                           (28,20)]
        self.circlepoint = [(self.pitch_width/2,self.pitch_height/2)]
        self.penaltypoint = [8,21]
        self.cornerpoints = [0,5,24,29]

class PitchTransformation:
    
    def __init__(self,source,target):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m,_ = cv2.findHomography(source,target)
    
    def dimension_coordinates_transformation(self, points) -> np.ndarray:
        points = points.reshape(-1,1,2).astype(np.float32)
        points = cv2.perspectiveTransform(points,self.m)
        return points.reshape(-1,2).astype(np.float32)