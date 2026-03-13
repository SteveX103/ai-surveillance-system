from database import db
from face_recognition_module import face_system
from datetime import datetime as dt
import configure as cfg
import os 
import cv2
import base64
from werkzeug.utils import secure_filename
from flask import Flask , Response , jsonify , request , render_template
from flask_cors import CORS

app = Flask(__name__, 
            template_folder ='../frontend/templates',
            static_folder = '../frontend/static')
CORS(app)
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(cfg.CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
    return camera
def generate_frames():
    cam = get_camera()
    while True:
        success , frame = cam.read()
        if not success:
            break
        processed_frame , detected_faces = face_system.detect_and_recognize_faces(frame)
        ret,buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
        
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/admin')
def admin():
    return render_template('admin.html')
@app.route('/register')
def register_page():
    return render_template('register.html')
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


