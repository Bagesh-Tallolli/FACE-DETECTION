import os
import re
import base64
import datetime
from typing import Optional, Tuple
from pathlib import Path
import face_recognition
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///face_recognition.db', echo=False)
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    face_encoding = Column(LargeBinary, nullable=False)
    registration_date = Column(String, nullable=False)

Base.metadata.create_all(engine)

# Constants
FACE_DISTANCE_THRESHOLD = 0.45  # More strict threshold (was 0.6)
MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_DURATION = 15 * 60  # 15 minutes in seconds
MIN_FACE_SIZE = (80, 80)  # Minimum face size for detection
FACE_DETECTION_CONFIDENCE = 0.9  # Minimum confidence for face detection

# Global storage for login attempts and lockouts
login_attempts = {}
lockout_times = {}

def cleanup_old_images():
    """Remove face images older than 24 hours"""
    try:
        current_time = datetime.datetime.now()
        faces_dir = Path('static/faces')
        if not faces_dir.exists():
            return
            
        for file_path in faces_dir.glob('*'):
            if file_path.is_file():
                file_age = current_time - datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.days >= 1:
                    file_path.unlink()
    except Exception as e:
        print(f"Error during cleanup: {e}")

def save_face_image(filename: str, b64data: str) -> Optional[str]:
    """Save face image from base64 data and return the file path"""
    try:
        if not b64data or ',' not in b64data:
            raise ValueError("Invalid base64 image data")
            
        header, data = b64data.split(',', 1)
        ext_match = re.search(r'image/(\w+);', header)
        if not ext_match:
            raise ValueError("Invalid image format")
            
        ext = ext_match.group(1)
        if ext.lower() not in ['jpeg', 'jpg', 'png']:
            raise ValueError("Unsupported image format")
            
        img_data = base64.b64decode(data)
        path = f'static/faces/{filename}.{ext}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            f.write(img_data)
        return path
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def get_face_encoding(image_path: str) -> Optional[np.ndarray]:
    """Extract face encoding from image with improved accuracy"""
    try:
        # Read image
        image = face_recognition.load_image_file(image_path)
        
        # Find face locations with more accurate CNN model
        face_locations = face_recognition.face_locations(image, model="cnn")
        if not face_locations:
            # Fallback to HOG model if CNN fails
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations:
                print("No face detected in image")
                return None
        
        # Get the largest face (closest to camera)
        if len(face_locations) > 1:
            # Sort by face size (largest first)
            face_locations.sort(key=lambda rect: (rect[2] - rect[0]) * (rect[1] - rect[3]), reverse=True)
        
        # Check minimum face size
        top, right, bottom, left = face_locations[0]
        face_width = right - left
        face_height = bottom - top
        if face_width < MIN_FACE_SIZE[0] or face_height < MIN_FACE_SIZE[1]:
            print(f"Face too small: {face_width}x{face_height} pixels")
            return None
            
        # Get face encoding with increased accuracy
        encodings = face_recognition.face_encodings(
            image, 
            [face_locations[0]], 
            num_jitters=5,  # Increased from 2
            model="large"  # Use large model for better accuracy
        )
        
        if not encodings:
            print("Could not compute face encoding")
            return None
            
        return encodings[0]
    except Exception as e:
        print(f"Error getting face encoding: {e}")
        return None

def verify_face(stored_encoding: bytes, login_image_path: str, threshold: float = FACE_DISTANCE_THRESHOLD) -> Tuple[bool, float]:
    """Verify face with improved accuracy checks"""
    try:
        if stored_encoding is None:
            print("[verify_face] Stored encoding is None")
            return False, 0.0
            
        # Get login face encoding
        login_encoding = get_face_encoding(login_image_path)
        if login_encoding is None:
            print("[verify_face] Could not extract login face encoding")
            return False, 0.0
            
        # Convert stored encoding from bytes
        stored_encoding_array = np.frombuffer(stored_encoding, dtype=np.float64)
        
        # Calculate face distance
        face_distance = face_recognition.face_distance([stored_encoding_array], login_encoding)[0]
        
        # Convert distance to similarity score (0-1, higher is better)
        similarity = 1 - face_distance
        
        # Add additional checks for very low similarity scores
        if similarity < 0.3:  # Extremely low similarity
            print(f"[verify_face] Very low similarity score: {similarity:.3f}")
            return False, similarity
            
        # Check if similarity exceeds threshold
        is_match = similarity >= (1 - threshold)
        print(f"[verify_face] Similarity score: {similarity:.3f}, Threshold: {1-threshold:.3f}, Match: {is_match}")
        
        return is_match, similarity
        
    except Exception as e:
        print(f"[verify_face] Error: {e}")
        return False, 0.0

def send_alert_email(to_email: str, image_path: str, attempt_details: str):
    """Send security alert email for failed login attempts"""
    if not os.getenv('SMTP_EMAIL') or not os.getenv('SMTP_PASSWORD'):
        print("Email configuration missing")
        return
        
    try:
        msg = EmailMessage()
        msg['Subject'] = '🚨 Security Alert - Unauthorized Login Attempt'
        msg['From'] = os.getenv('SMTP_EMAIL')
        msg['To'] = to_email
        
        content = f"""
        ⚠️ SECURITY ALERT ⚠️
        
        An unauthorized login attempt was detected on your account.
        
        Details:
        - Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Location: Remote Access
        - Status: Account Temporarily Locked
        - Reason: Multiple Failed Face Recognition Attempts
        
        Additional Details:
        {attempt_details}
        
        For security reasons, your account has been temporarily locked.
        
        If this wasn't you, please:
        1. Change your password immediately
        2. Enable two-factor authentication if available
        3. Contact support if suspicious activity continues
        
        Stay Safe! 🔒
        """
        
        msg.set_content(content)
        
        # Attach the unauthorized attempt image
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
                msg.add_attachment(img_data, maintype='image', 
                                 subtype=image_path.split('.')[-1],
                                 filename='unauthorized_attempt.jpg')
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(os.getenv('SMTP_EMAIL'), os.getenv('SMTP_PASSWORD'))
            smtp.send_message(msg)
    except Exception as e:
        print(f"Failed to send alert email: {e}")

@app.before_request
def before_request():
    """Cleanup old images before processing requests"""
    cleanup_old_images()

@app.route('/')
def home():
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration with face recognition"""
    if request.method == 'GET':
        return render_template('register.html')
        
    db = SessionLocal()
    try:
        # Get form data
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        face_data = request.form.get('face_data', '')
        
        if not all([email, password, face_data]):
            return "All fields are required", 400
            
        # Check if user exists
        if db.query(User).filter_by(email=email).first():
            return "Email already registered", 400
            
        # Save and process face image
        img_path = save_face_image(email.replace("@", "_").replace(".", "_"), face_data)
        if not img_path:
            return "Error saving face image", 400
            
        try:
            # Get face encoding
            face_encoding = get_face_encoding(img_path)
            if face_encoding is None:
                os.unlink(img_path)
                return "No face detected or could not process face", 400
                
            # Create user
            user = User(
                email=email,
                password=password,  # In production, hash the password
                face_encoding=face_encoding.tobytes(),
                registration_date=datetime.datetime.now().isoformat()
            )
            
            db.add(user)
            db.commit()
            return redirect('/login')
            
        except Exception as e:
            db.rollback()
            if os.path.exists(img_path):
                os.unlink(img_path)
            print(f"Registration error: {e}")
            return "Registration failed", 400
            
    except Exception as e:
        print(f"Registration error: {e}")
        return "Registration failed", 400
    finally:
        db.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login with face recognition"""
    if request.method == 'GET':
        return render_template('login.html')
        
    db = SessionLocal()
    try:
        # Get form data
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        face_data = request.form.get('face_data', '')
        
        if not all([email, password, face_data]):
            return "All fields are required", 403
            
        # Check lockout
        if email in lockout_times:
            lockout_time = lockout_times[email]
            if datetime.datetime.now().timestamp() < lockout_time:
                remaining_time = int(lockout_time - datetime.datetime.now().timestamp())
                return render_template('locked.html', remaining_time=remaining_time), 403
                
        # Verify credentials
        user = db.query(User).filter_by(email=email, password=password).first()
        if not user:
            return "Invalid email or password", 403
            
        try:
            # Save and process login attempt image
            img_path = save_face_image(f"{email}_login_{int(datetime.datetime.now().timestamp())}", face_data)
            if not img_path:
                return "Error processing face image", 403
                
            # Verify face
            is_match, similarity = verify_face(user.face_encoding, img_path)
            
            if is_match:
                # Successful login
                login_attempts[email] = 0
                if os.path.exists(img_path):
                    os.unlink(img_path)
                return f"Access Granted (Confidence: {similarity:.2f})"
            else:
                # Failed login attempt
                login_attempts[email] = login_attempts.get(email, 0) + 1
                attempt_details = f"Similarity score: {similarity:.3f}"
                
                if login_attempts[email] >= MAX_LOGIN_ATTEMPTS:
                    # Lock account
                    lockout_times[email] = datetime.datetime.now().timestamp() + LOCKOUT_DURATION
                    send_alert_email(user.email, img_path, attempt_details)
                    return render_template('locked.html', remaining_time=LOCKOUT_DURATION), 403
                    
                return f"Face verification failed (Attempt {login_attempts[email]}/{MAX_LOGIN_ATTEMPTS})", 403
                
        except Exception as e:
            print(f"Login error: {e}")
            return "Face verification failed", 403
            
    except Exception as e:
        print(f"Login error: {e}")
        return "Login failed", 403
    finally:
        db.close()

if __name__ == '__main__':
    os.makedirs('static/faces', exist_ok=True)
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        ssl_context='adhoc' if os.getenv('ENABLE_HTTPS', '0') == '1' else None
    ) 