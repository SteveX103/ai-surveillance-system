import os
from datetime import datetime as dt

Base_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWN_FACES_DIR = os.path.join(Base_DIR,'..','known_faces')
UNKNOWN_FACES_DIR = os.path.join(Base_DIR,'..','unknown_faces')

MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = "ai_surveillance"
COLLECTION_LOGS = "detection_logs"
COLLECTION_KNOWN_FACES = "known_faces"

CAMERA_INDEX = 0
FRAME_WIDTH = 640   
FRAME_HEIGHT = 480

UNKNOWN_FACE_CAPTTURE_INTERVAL = 5  
FACE_DETECTION_CONFIDENCE = 0.7

def get_today_folder():
    today = dt.now().strftime("%d-%m-%y")
    folder_path = os.path.join(UNKNOWN_FACES_DIR, today)
    os.makedirs(folder_path, exist_ok = True)
    return folder_path                             