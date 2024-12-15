import cv2
import numpy as np
import supervision as sv
from pitch_dimension import SoccerPitch

Pitch_Measures = SoccerPitch()
Pitch_Vertices = Pitch_Measures.vertices
pitch_scale = 0.1
pitch_color_lines = sv.Color.WHITE.as_bgr()
pitch_width = int(Pitch_Measures.pitch_width * pitch_scale)
pitch_height = int(Pitch_Measures.pitch_height * pitch_scale)
pitch_radius = int(Pitch_Measures.pitch_center_radius * pitch_scale)
pitch_corner_radius = int(Pitch_Measures.pitch_corner_radius * pitch_scale)
pitch_padding = 30 # 30px
    
def pitch_drawing_cv2():
    
    newPitch = np.zeros((pitch_height+pitch_padding,pitch_width+pitch_padding,3), np.uint8)
    
    newPitch[:,:,0] = 41
    newPitch[:,:,1] = 41
    newPitch[:,:,2] = 41
    
    
    for initial,final in Pitch_Measures.linepoints:
        pt1=(int(Pitch_Vertices[initial][0] * pitch_scale + pitch_padding/2), int(Pitch_Vertices[initial][1] * pitch_scale + pitch_padding/2))
        pt2=(int(Pitch_Vertices[final][0] * pitch_scale + pitch_padding/2), int(Pitch_Vertices[final][1] * pitch_scale + pitch_padding/2))
        draw_linep2p(image=newPitch,pt1=pt1,pt2=pt2,color=pitch_color_lines)
        
    for point in Pitch_Measures.penaltypoint:
        pt1=(int(Pitch_Vertices[point][0] * pitch_scale + pitch_padding/2), int(Pitch_Vertices[point][1] * pitch_scale + pitch_padding/2))
        draw_point(image=newPitch,pt1=pt1,color=pitch_color_lines)
        if point % 2 == 0:
            draw_ellipse(image=newPitch,pt1=pt1,radius=pitch_radius,angle=0,sAngle=37.5,eAngle=142.5,color=pitch_color_lines)
        else:
            draw_ellipse(image=newPitch,pt1=pt1,radius=pitch_radius,angle=0,sAngle=-142.5,eAngle=-37.5,color=pitch_color_lines)
    
    evenAngle=0
    oddAngle=0  
    for point in Pitch_Measures.cornerpoints:
        pt1=(int(Pitch_Vertices[point][0] * pitch_scale + pitch_padding/2), int(Pitch_Vertices[point][1] * pitch_scale + pitch_padding/2))
        if point % 2 == 0:
            draw_ellipse(image=newPitch,pt1=pt1,radius=pitch_corner_radius,angle=evenAngle,sAngle=0,eAngle=90,color=pitch_color_lines)
            evenAngle = -90
        else:
            draw_ellipse(image=newPitch,pt1=pt1,radius=pitch_corner_radius,angle=oddAngle,sAngle=-180,eAngle=-270,color=pitch_color_lines)
            oddAngle = 90 
    
    pt3=(int(Pitch_Measures.circlepoint[0][0]*pitch_scale + pitch_padding/2) ,int(Pitch_Measures.circlepoint[0][1]*pitch_scale + pitch_padding/2))
    draw_centercircle(image=newPitch,pt1=pt3,radius=pitch_radius,color=pitch_color_lines)
    
    cv2.imwrite("C:/Football Analysis/pitch.png",newPitch)
    
    return newPitch
    
def draw_linep2p(image,pt1,pt2,color):
    cv2.line(img=image,pt1=pt1,pt2=pt2,color=color,thickness=4)
            
def draw_centercircle(image,pt1,radius,color):
    cv2.circle(img=image,center=pt1,radius=radius,color=color,thickness=4)
    draw_point(image=image,pt1=pt1,color=color)
        
def draw_point(image,pt1,color):
    cv2.circle(img=image,center=pt1,radius=3,color=color,thickness=-1)
        
def draw_ellipse(image,pt1,radius,sAngle,angle,eAngle,color):
    axes = (radius,radius)
    cv2.ellipse(img=image,center=pt1,axes=axes,angle=angle,startAngle=sAngle,endAngle=eAngle,color=color,thickness=3)
    
def draw_player_on_pitch(image,point,radius,color):
    # Acabar esta função
    
    #point = int(point[0]*pitch_scale)
    
    cv2.circle(img=image,center=point,radius=int(radius),color=color)
    
def draw_ball_on_pitch(image,point,radius,color):
    # Acabar esta função
    
    #point = int(point[0]*pitch_scale)
    
    cv2.circle(img=image,center=point,radius=int(radius),color=color)