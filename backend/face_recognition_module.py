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
        print("Loading known faces from database...")
        known_faces = db.get_all_knoewn_faces()
        self.known_face_encodings = []
        self.known_face_names = []
        for face_data in known_faces:
            try:
                encoding = np.array(face_data['face_encoding'])
                self.known_faces_encodings.append(encoding)
                self.known_faces_names.append(face_data['name'])
                print(f"Loaded known face: {face_data['name']}")
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
            