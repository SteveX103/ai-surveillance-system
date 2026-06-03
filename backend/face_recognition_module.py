import cv2
import face_recognition
import numpy as np
from datetime import datetime, timedelta
import os
from PIL import Image
import configure as config
from database import db

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_unknown_capture = {}  
        self.process_this_frame = True
        
        self.load_known_faces()
    
    def load_known_faces(self):
        print("Loading known faces from database...")
        known_faces = db.get_all_known_faces()
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        for face_data in known_faces:
            try:
                encoding = np.array(face_data['face_encoding'])
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(face_data['name'])
                print(f" Loaded: {face_data['name']}")
            except Exception as e:
                print(f" Error loading face {face_data.get('name', 'Unknown')}: {e}")
        
        print(f" Total {len(self.known_face_names)} known faces loaded!")
    
    
    def _load_and_convert_image(self, image_path):
        try:
            pil_image = Image.open(image_path)

            pil_image = pil_image.convert('RGB') # Force RGB
            image = np.array(pil_image, dtype=np.uint8)# Convert to numpy uint8

            if len(image.shape) != 3:
                raise Exception(f"Invalid image dimensions: {image.shape}")

            if image.shape[2] != 3:
                raise Exception(f"Invalid channels: {image.shape[2]}")

            return image

        except Exception as e:
            raise Exception(f"Failed to load image properly: {e}")


    def register_new_face(self, image_path, name):
        """Naya face register karo with improved error handling"""
        try:
            print(f"Registering face for: {name}")
            print(f"Image path: {image_path}")
            
            
            if not os.path.exists(image_path):
                return False, "Image file not found"
            
            
            try:
                image = self._load_and_convert_image(image_path)
                print(f" Image loaded successfully: {image.shape}")
            except Exception as load_error:
                print(f" Image load error: {load_error}")
                return False, f"Error loading image: {str(load_error)}"
            
            
            if image is None or len(image.shape) != 3:
                return False, "Invalid image format"
            
            if image.shape[2] != 3:
                return False, "Image must be RGB (3 channels)"
            
            print(f"Detecting faces in image...")
            

            
            try:
                
                image = image.astype(np.uint8)
                image = np.ascontiguousarray(image)

                
                face_encodings = face_recognition.face_encodings(image)

                print(f" Found {len(face_encodings)} face(s)")
            except Exception as enc_error:
                print(f" Encoding error: {enc_error}")
                return False, f"Error detecting face: {str(enc_error)}"
            
            if len(face_encodings) == 0:
                return False, "No face detected in image. Please use a clear photo with visible face."
            
            if len(face_encodings) > 1:
                return False, f"Multiple faces detected ({len(face_encodings)}). Please use image with single face."
            
            face_encoding = face_encodings[0]
            print(f" Face encoding created successfully")
            
            
            print(f" Saving to database...")
            result = db.register_known_face(name, face_encoding, image_path)
            
            if result:
                
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                print(f" Face registered successfully for {name}")
                return True, f"Face registered successfully for {name}"
            else:
                return False, "Database error while saving"
                
        except Exception as e:
            error_msg = f"Error in register_new_face: {str(e)}"
            print(f" {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg
    def detect_and_recognize_faces(self, frame):

        if frame is None:
            return frame, []

        frame = np.array(frame, dtype=np.uint8)# FORCE uint8

        
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)# Remove alpha channel if exists

        
        frame = np.ascontiguousarray(frame)# Ensure contiguous memory

        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        
        rgb_small_frame = np.array(rgb_small_frame, dtype=np.uint8)
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame)

        # DEBUG
        print("FRAME INFO:")
        print(rgb_small_frame.dtype)
        print(rgb_small_frame.shape)
        print(rgb_small_frame.flags)

        if not hasattr(self, 'face_names'):
            self.face_names = []
            self.face_locations = []

        if self.process_this_frame:
            try:
                face_locations = face_recognition.face_locations(rgb_small_frame)

                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame,
                    face_locations
                )

                self.face_names = []
                self.face_locations = face_locations

                for face_encoding in face_encodings:

                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        face_encoding,
                        tolerance=config.FACE_DETECTION_CONFIDENCE
                    )

                    name = "Unknown"

                    if len(self.known_face_encodings) > 0:

                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings,
                            face_encoding
                        )

                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)

                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index]
                                db.log_known_detection(name)

                    self.face_names.append(name)

            except Exception as e:
                print(f" Detection error: {e}")

        self.process_this_frame = not self.process_this_frame

        
        detected_faces = []
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            
            if name == "Unknown":
                color = (0, 0, 255)# Red for unknown
                self._handle_unknown_face(frame, (top, right, bottom, left))
            else:
                color = (0, 255, 0)# Green for known
            
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
           
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            detected_faces.append({
                'name': name,
                'location': (top, right, bottom, left)
            })
        
        return frame, detected_faces
    
    def _handle_unknown_face(self, frame, location):
        current_time = datetime.now()
        location_key = str(location) 
        
        if location_key in self.last_unknown_capture:
            time_diff = (current_time - self.last_unknown_capture[location_key]).total_seconds()
            if time_diff < config.UNKNOWN_FACE_CAPTURE_INTERVAL:
                return  
        
        top, right, bottom, left = location
        
        padding = 20
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(frame.shape[0], bottom + padding)
        right = min(frame.shape[1], right + padding)
        
        face_image = frame[top:bottom, left:right]
        
        if face_image.size == 0:
            return
        
        today_folder = config.get_today_folder()
        timestamp = current_time.strftime("%H-%M-%S")
        filename = f"unknown_{timestamp}.jpg"
        filepath = os.path.join(today_folder, filename)
        
        try:
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_image_rgb)
            pil_image.save(filepath, 'JPEG', quality=95)
            
            print(f" Unknown face captured: {filepath}")
            
            db.log_unknown_detection(filepath, current_time)
            
            self.last_unknown_capture[location_key] = current_time
            
        except Exception as e:
            print(f"❌ Error saving unknown face: {e}")

face_system = FaceRecognitionSystem()