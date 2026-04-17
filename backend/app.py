from database import db
from face_recognition_module import FaceRecognitionSystem
face_system = FaceRecognitionSystem()
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
           cam.release()
           break
        processed_frame , detected_faces = face_system.detect_and_recognize_faces(frame)
        if processed_frame is None:
            continue

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

@app.route('/api/stats')
def stats():
    try:
        today_count= db.get_unknown_count_today()
        total_count=db.get_unknown_count_total()
        recent_detections= db.get_recent_detections(limit = 10)
        for detection in recent_detections:
            detection['_id'] = str(detection['_id'])

        return jsonify({
                'success' : True,
                'today_count': today_count,
                'total_count': total_count,
                'recent_detections': recent_detections,
                'known_faces_count': len(face_system.known_face_names)
            })
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return jsonify({'success': False, 
                        'error': 'Failed to fetch stats'}),500
@app.route('/api/known_faces')
def get_known_faces():
    try:
        faces=db.get_all_known_faces()
        for face in faces:
            face['_id'] = str(face['_id'])
            if 'face_encoding' in face:
                del face['face_encoding']

        return jsonify({
                    'success': True,
                    'faces':faces
            })
    except Exception as e:
        print(f"Error fetching known faces: {e}")
        return jsonify({
            'success': False, 
            'error': 'Failed to fetch known faces'}),500
@app.route('/api/register_face', methods = ['POST'])
def register_face():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided',
            }),400
        if 'name' not in request.form:
            return jsonify({
                'success': False,
                'error': 'No name provided'
            }),400
        image = request.files['image']
        name = request.form['name']
        if image.filename == '':
            return jsonify({
                'success' :False,
                'error': 'No selected file'
            }),400
        filename = secure_filename(f"{name}_{dt.now().strftime('%Y%m%d%H%M%S')}.jpg")
        filepath = os.path.join(cfg.KNOWN_FACES_DIR,filename)
        image.save(filepath)
        success , message = face_system.register_new_face(filepath,name)
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e :
        return jsonify({
            'success' : False,
            'message' : str(e)
        }),500
@app.route('/api/delete_face/<name>', methods = ['DELETE'])
def delete_face(name):
    try:
        success= db.delete_known_face(name)
        if success:
            face_system.load_known_faces()
            return jsonify({
                'success': True,
                'message': f'Face {name} deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Face not found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }),500
@app.route('/api/capture_from_webcam', methods=['POST'])
def capture_from_webcam():
    try:
        data = request.get_json()
        image_data = data.get('image')
        name= data.get('name')
        if not image_data or not name:
            return jsonify({
                'success': False,
                'error': 'Image data and name are required'
            }),400
        image_data = image_data.split(',')[1]
        image_bytes=base64.b64decode(image_data)
        filename = secure_filename(f"{name}_{dt.now().strftime('%Y%m%d%H%M%S')}.jpg")
        filepath = os.path.join(cfg.KNOWN_FACES_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        success, message = face_system.register_new_face(filepath, name)
        return jsonify({
            'success': success,
            'message': message
        })  
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }),500

@app.route('/release_camera')
def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return "Camera released"

if __name__ == '__main__':
    print("Starting AI Surveillance System...")
    print(f"Camera Index: {cfg.CAMERA_INDEX}")
    print(f"Database: {cfg.DB_NAME}")
    print(f"Server: http://localhost:5000")
    print("\nAccess Points:")
    print("   - Live Feed: http://localhost:5000")
    print("   - Admin Panel: http://localhost:5000/admin")
    print("   - Register Face: http://localhost:5000/register")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
