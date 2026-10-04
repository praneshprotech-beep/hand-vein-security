from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key"
db_file = "hand_vein_db.json"
logs_file = "auth_logs.json"

# Hard-coded admin credentials for demonstration purposes
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Load ResNet50 model without top layers for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten().tolist()

def save_user(name, passport_id, nationality, fingerprint_features):
    if os.path.exists(db_file):
        with open(db_file, "r") as file:
            data = json.load(file)
    else:
        data = {}
    
    data[name] = {
        "passport_id": passport_id,
        "nationality": nationality,
        "features": fingerprint_features
    }
    
    with open(db_file, "w") as file:
        json.dump(data, file)

def authenticate_user(name, fingerprint_features):
    if not os.path.exists(db_file):
        return False
    
    with open(db_file, "r") as file:
        data = json.load(file)
    
    if name in data:
        stored_features = np.array(data[name]["features"])
        input_features = np.array(fingerprint_features)
        similarity = np.linalg.norm(stored_features - input_features)
        return similarity < 10.0  # Threshold for authentication
    return False

def log_auth_attempt(user, status, details=""):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "status": status,
        "details": details,
        "ip": request.remote_addr
    }
    if os.path.exists(logs_file):
        with open(logs_file, "r") as file:
            logs = json.load(file)
    else:
        logs = []
    logs.append(log_entry)
    with open(logs_file, "w") as file:
        json.dump(logs, file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        passport_id = request.form.get('passport_id', '').strip()
        nationality = request.form.get('nationality', '').strip()
        fingerprint = request.files.get('fingerprint')
        filename = secure_filename(fingerprint.filename) if fingerprint else ''
        home_action = {'label': 'Back to Home', 'url': url_for('home')}

        if not name or not passport_id or not nationality or not filename:
            return render_template(
                'result.html',
                title='Registration failed',
                message='Please complete all fields and choose a valid fingerprint image.',
                actions=[
                    {'label': 'Try Again', 'url': url_for('register')},
                    home_action,
                ],
                success=False,
            )

        img_path = os.path.join('static', 'uploads', filename)
        try:
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            fingerprint.save(img_path)
            features = extract_features(img_path)
            save_user(name, passport_id, nationality, features)
        except Exception:
            app.logger.exception('Registration failed')
            return render_template(
                'result.html',
                title='Registration failed',
                message='We could not process that image. Please try again with a supported image file.',
                actions=[
                    {'label': 'Try Again', 'url': url_for('register')},
                    home_action,
                ],
                success=False,
            )

        return render_template(
            'result.html',
            title='Registration successful',
            message='Your details and fingerprint have been registered.',
            actions=[
                home_action,
                {'label': 'Continue to Login', 'url': url_for('login')},
            ],
            success=True,
        )
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        passport_id = request.form.get('passport_id', '').strip()
        fingerprint = request.files.get('fingerprint')
        filename = secure_filename(fingerprint.filename) if fingerprint else ''
        home_action = {'label': 'Back to Home', 'url': url_for('home')}

        if not name or not passport_id or not filename:
            return render_template(
                'result.html',
                title='Login failed',
                message='Please enter your name and passport ID, and choose a fingerprint image.',
                actions=[
                    {'label': 'Try Again', 'url': url_for('login')},
                    home_action,
                ],
                success=False,
            )

        img_path = os.path.join('static', 'uploads', filename)
        try:
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            fingerprint.save(img_path)
            features = extract_features(img_path)
        except Exception:
            app.logger.exception('Login image processing failed')
            return render_template(
                'result.html',
                title='Login failed',
                message='We could not process that image. Please try again with a supported image file.',
                actions=[
                    {'label': 'Try Again', 'url': url_for('login')},
                    home_action,
                ],
                success=False,
            )

        if authenticate_user(name, features):
            session['user'] = name
            log_auth_attempt(name, "Success", "User authenticated successfully.")
            return redirect(url_for('dashboard'))

        log_auth_attempt(name, "Failure", "Authentication failed.")
        return render_template(
            'result.html',
            title='Login failed',
            message='We could not verify your fingerprint. Check your details and try again.',
            actions=[
                {'label': 'Try Again', 'url': url_for('login')},
                {'label': 'Register', 'url': url_for('register')},
                home_action,
            ],
            success=False,
        )
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    if os.path.exists(db_file):
        with open(db_file, "r") as file:
            data = json.load(file)
        if user in data:
            user_data = data[user]
            return render_template('profile.html', name=user, passport_id=user_data["passport_id"], nationality=user_data["nationality"])
    return "Profile not found!"

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# Admin routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['is_admin'] = True
            session['user'] = ADMIN_USERNAME
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template(
                'result.html',
                title='Admin login failed',
                message='The username or password was not recognized. Please try again.',
                actions=[
                    {'label': 'Try Again', 'url': url_for('admin_login')},
                    {'label': 'Back to Home', 'url': url_for('home')},
                ],
                success=False,
            )
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    # Load all registered users
    if os.path.exists(db_file):
        with open(db_file, "r") as file:
            users = json.load(file)
    else:
        users = {}
    # Load logs
    if os.path.exists(logs_file):
        with open(logs_file, "r") as file:
            logs = json.load(file)
    else:
        logs = []
    return render_template('admin_dashboard.html', users=users, logs=logs)

if __name__ == '__main__':
    if not os.path.exists("static/uploads"):
        os.makedirs("static/uploads")
    app.run(debug=True)
