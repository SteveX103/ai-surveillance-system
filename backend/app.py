from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import os
from datetime import datetime
import base64
from werkzeug.utils import secure_filename
import threading
from PIL import Image
import io
import configure as config
from face_recognition_module import face_system
from database import db

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

camera = None
camera_lock = threading.Lock()
active_streams = 0

def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(config.CAMERA_INDEX)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            print(" Camera initialized")
        return camera

def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            print(" Camera released")

def generate_frames():
    global active_streams
    
    try:
        active_streams += 1
        cam = get_camera()
        
        while True:
            with camera_lock:
                if cam is None or not cam.isOpened():
                    break
                    
                success, frame = cam.read()
                if not success:
                    break
            
            
            processed_frame, detected_faces = face_system.detect_and_recognize_faces(frame)
            
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    except Exception as e:
        print(f" Stream error: {e}")
    finally:
        active_streams -= 1
        if active_streams == 0:
            release_camera()

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
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/release_camera', methods=['POST'])
def release_camera_endpoint():
    try:
        release_camera()
        return jsonify({
            'success': True,
            'message': 'Camera released successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/stats')
def get_stats():
    try:
        today_count = db.get_unknown_count_today()
        total_count = db.get_unknown_count_total()
        recent_detections = db.get_recent_detections(limit=10)
        
        for detection in recent_detections:
            detection['_id'] = str(detection['_id'])
        
        return jsonify({
            'success': True,
            'today_count': today_count,
            'total_count': total_count,
            'recent_detections': recent_detections,
            'known_faces_count': len(face_system.known_face_names)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/known_faces')
def get_known_faces():
    """Saare known faces ki list"""
    try:
        faces = db.get_all_known_faces()
        
        for face in faces:
            face['_id'] = str(face['_id'])
            if 'face_encoding' in face:
                del face['face_encoding']
        
        return jsonify({
            'success': True,
            'faces': faces
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/register_face', methods=['POST'])
def register_face():
    try:
        print(" Received face registration request (file upload)")
        
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image provided'
            }), 400
        
        if 'name' not in request.form:
            return jsonify({
                'success': False,
                'message': 'No name provided'
            }), 400
        
        image_file = request.files['image']
        name = request.form['name'].strip()
        
        if not name:
            return jsonify({
                'success': False,
                'message': 'Name cannot be empty'
            }), 400
        
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No image selected'
            }), 400
        
        print(f" Name: {name}")
        print(f" File: {image_file.filename}")
        
        os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)
        
        filename = secure_filename(f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        filepath = os.path.join(config.KNOWN_FACES_DIR, filename)
        
        try:
            image_data = image_file.read()
            
            pil_image = Image.open(io.BytesIO(image_data))
            
            if pil_image.mode != 'RGB':
                print(f"🔄 Converting from {pil_image.mode} to RGB")
                pil_image = pil_image.convert('RGB')
            
            pil_image.save(filepath, 'JPEG', quality=95)
            print(f" Image saved: {filepath}")
            
        except Exception as save_error:
            print(f" Error saving image: {save_error}")
            return jsonify({
                'success': False,
                'message': f'Error saving image: {str(save_error)}'
            }), 500
        
        success, message = face_system.register_new_face(filepath, name)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify({
                'success': False,
                'message': message
            }), 400
        
    except Exception as e:
        print(f"❌ Registration error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/delete_face/<name>', methods=['DELETE'])
def delete_face(name):
    try:
        success = db.delete_known_face(name)
        
        if success:
            face_system.load_known_faces() 
            return jsonify({
                'success': True,
                'message': f'{name} deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Face not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/capture_from_webcam', methods=['POST'])
def capture_from_webcam():
    try:
        print(" Received webcam capture request")
        
        data = request.get_json()
        image_data = data.get('image')
        name = data.get('name', '').strip()
        
        if not image_data or not name:
            return jsonify({
                'success': False,
                'message': 'Image and name required'
            }), 400
        
        print(f" Name: {name}")
        
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            print(f" Base64 decoded: {len(image_bytes)} bytes")
            
        except Exception as decode_error:
            print(f" Base64 decode error: {decode_error}")
            return jsonify({
                'success': False,
                'message': 'Invalid image data'
            }), 400
        
        os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)
        
        filename = secure_filename(f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        filepath = os.path.join(config.KNOWN_FACES_DIR, filename)
        
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
           
            if pil_image.mode != 'RGB':
                print(f"🔄 Converting from {pil_image.mode} to RGB")
                pil_image = pil_image.convert('RGB')
            
            pil_image.save(filepath, 'JPEG', quality=95)
            print(f"✅ Image saved: {filepath}")
            
        except Exception as save_error:
            print(f" Error saving image: {save_error}")
            return jsonify({
                'success': False,
                'message': f'Error saving image: {str(save_error)}'
            }), 500
       
        success, message = face_system.register_new_face(filepath, name)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify({
                'success': False,
                'message': message
            }), 400
        
    except Exception as e:
        print(f" Webcam capture error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Starting AI Surveillance System...")
    print(f" Camera Index: {config.CAMERA_INDEX}")
    print(f" Database: {config.DB_NAME}")
    print(f" Server: http://localhost:5000")
    print("\n Access Points:")
    print("   - Live Feed: http://localhost:5000")
    print("   - Admin Panel: http://localhost:5000/admin")
    print("   - Register Face: http://localhost:5000/register")
    print("\n Notes")
    print("   - Close live feed tab before using webcam registration")
    print("   - Use clear, well-lit photos for best results")
    print("   - One face per image only")
    print("\n" + "="*60 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    finally:
        release_camera()
        print("\nServer stopped and camera released")