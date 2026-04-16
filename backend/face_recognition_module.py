import os 
import cv2
import configure as cfg
import numpy as np
import face_recognition as fr
from datetime import datetime as dt , timedelta
from database import db
from PIL import Image

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_unknown_capture = {}
        self.process_this_frame = True
        self.load_known_faces()

    def load_known_faces(self):
        print("Loading known faces from Database....")
        known_faces = db.get_all_known_faces()
        self.known_face_encodings = []
        self.known_face_names = []
        for face_data in known_faces:
            try:
                encoding = np.array(face_data['face_encoding'])
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(face_data['name'])
                print(f"Loaded known facew: {face_data['name']}")
            except Exception as e:
                print(f"Error loading known face {face_data['name']}: {e}")

        print(f"Total known faces loaded: {len(self.known_face_encodings)}")

    def register_new_face(self,image_path, name):
        try:
            image = fr.load_image_file(image_path)
            face_encodings = fr.face_encodings(image)
            if len(face_encodings) == 0:
                print("No face in the image")
                return False
            if len(face_encodings) > 1:
                print("many faces in the image")
                return False
            face_encodings = face_encodings[0]
            result = db.register_known_face(name,face_encodings , image_path)
            if result:
                self.known_face_encodings.append(face_encodings)
                self.known_face_names.append(name)
                print(f"Face registered succesfully for : {name}")
                return True
            else:
                print("Failed to register new face.")
                return False
        except Exception as e  :
            print(f"Error registering new face: {e}")
            return False 
            
    def detect_and_recognize_faces(self,frame):
        if frame is None:
            return None, []
        small_frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
        if small_frame is None or small_frame.dtype != 'unit8':
            return frame,[]
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small_frame = rgb_small_frame.astype('uint8')
        if self.process_this_frame:
            face_locations = fr.face_locations(rgb_small_frame)
            face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
            self.face_names=[]
            self.face_locations=face_locations
            for face_encoding in face_encodings:
                matches = fr.compare_faces(self.known_face_encodings,face_encoding,tolerance= cfg.FACE_DETECTION_CONFIDENCE)
                name="Unknown"
                face_distance= fr.face_distance(self.known_face_encodings,face_encoding)
                if len(face_distance) > 0:
                    best_match_index = np.argmin(face_distance)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        db.log_known_detection(name)
                    self.face_names.append(name)
            self.process_this_frame = not self.process_this_frame
            detected_faces = []
            for ( top,right,bottom,left), name in zip(self.face_locations,self.face_names):
                top *=4
                right *=4
                bottom *=4
                left *=4
                if name == "Unknown":
                    color = (0,0,255)
                    self.handle_unknown_face(frame, (top,right,bottom,left))
                else:
                    color = (0,255,0)

                cv2.rectangle(frame,(left,top),(right,bottom), color,2)
                cv2.rectangle(frame,(left,bottom - 35),(right,bottom), color,cv2.FILLED)
                font = cv2.FONT_ITALIC
                cv2.putText(frame,name,(left+6 ,bottom-6), font,0.6,(255,255,255),1)
                detected_faces.append({
                    "name": name,
                    "location": (top,right,bottom,left)
                })

            return frame , detected_faces
    def handle_unknown_faces(self,frame,location):
        current_time = dt.now()
        location_key = str(location)
        if location_key in self.last_unknown_capture:
            time_diff = (current_time - self.last_unknown_capture[location_key]).total_seconds()
            if time_diff< cfg.UNKNOWN_FACE_CAPTTURE_INTERVAL:
                return
        top,right,bottom,left = location
        face_image = frame[top:bottom, left:right]
        today_folder  = cfg.get_today_folder()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        filename = f"Unknown_{timestamp}.jpg"
        filepath = os.path.join(today_folder,filename)
        try:
            cv2.imwrite(filepath , face_image)
            print(f"Captured unknown facee at {timestamp}")
            db.log_unknown_detection(filepath,current_time)
            self.last_unknown_capture[location_key] = current_time
        except Exception as e:      
            print(f"Error saving unknown face image: {e}")

face_recognition_system = FaceRecognitionSystem()

        
