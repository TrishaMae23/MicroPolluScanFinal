from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import os
import cv2
import numpy as np
import supervision as sv
from datetime import date, datetime
from ultralytics import YOLO
from sqlalchemy.orm import sessionmaker, scoped_session

# Initialize the app 
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'  # Consider using environment variables

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

# Detection result model
class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(100), nullable=False)
    station = db.Column(db.String(100), nullable=False)
    beads = db.Column(db.Integer, nullable=False, default=0)
    fragments = db.Column(db.Integer, nullable=False, default=0)
    fibers = db.Column(db.Integer, nullable=False, default=0)
    count = db.Column(db.Integer, nullable=False, default=0)
    date_created = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)  # Store date and time

    def to_json(self):
        return {
            "id": self.id,
            "filename": self.file_name,
            "station": self.station,
            "beads": self.beads,
            "fragments": self.fragments,
            "fibers": self.fibers,
            "count": self.count,
            "dateCaptured": self.date_created.strftime('%Y-%m-%d'),  # Format date for JSON
        }

# Forms
class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Register")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError("That username already exists. Please choose a different one.")

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")

# YOLO Model Initialization
model = YOLO("best.pt")

# Box annotator for bounding boxes
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

# Zone for polygon detection
ZONE_POLYGON = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
frame_width, frame_height = 1000, 1000
camera = cv2.VideoCapture()  
camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Define the frame dimensions
frame_width = 1920  
frame_height = 1080  

# Zone polygon setup
scaled_zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)

# Create a PolygonZone object with the scaled polygon and frame resolution
zone = sv.PolygonZone(
    polygon=scaled_zone_polygon,
    frame_resolution_wh=(frame_width, frame_height)
)

# Initialize the PolygonZoneAnnotator for visualizing the zone
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.white(),
    thickness=2,
    text_thickness=4,
    text_scale=2
)

latest_frame = None
captured_frame_path = "static/temp/captured_image.jpg"

# Initialize a session factory
session_factory = None

@login_manager.user_loader
def load_user(user_id):
    if session_factory is None:
        raise RuntimeError("Session factory is not initialized.")
    session = session_factory()
    try:
        return session.get(User, int(user_id))
    finally:
        session.close()

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('detection'))
        else:
            flash('Login Unsuccessful. Please check username and password.', 'danger')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/detection', methods=['GET', 'POST'])
@login_required
def detection():
    return render_template('detection.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def generate_frames():
    global latest_frame
    while True:
        success, frame = camera.read()
        if not success:
            break

        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        latest_frame = frame
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/capture_frame", methods=["POST"])
def capture_frame():
    global latest_frame
    if latest_frame is not None:
        success, frame = camera.read()

        # YOLOv8 detection on the frame
        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)

        # Initialize counts for each type of microplastic
        counts = {
            "beads": 0,
            "fragments": 0,
            "fibers": 0,
        }

        # Define a confidence threshold
        confidence_threshold = 0.0  # Adjust this value as needed

        # Count detected microplastics based on their class
        for _, confidence, class_id, _ in detections:
            if confidence >= confidence_threshold:  # Check confidence
                class_name = model.model.names[class_id]
                if class_name == "bead":
                    counts["beads"] += 1
                elif class_name == "fragment":
                    counts["fragments"] += 1
                elif class_name == "fiber":
                    counts["fibers"] += 1

        total_count = sum(counts.values())

        # Create labels for detections above the confidence threshold
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections if confidence >= confidence_threshold
        ]

        # Annotate the frame with bounding boxes and labels
        frame = box_annotator.annotate(
            scene=frame, detections=detections, labels=labels
        )

        # Trigger zone for detections within the defined polygon
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        # Save the captured frame
        cv2.imwrite(captured_frame_path, frame)
        
        return jsonify(
            {
                "status": "success",
                "image_url": f"/{captured_frame_path}?t={os.path.getmtime(captured_frame_path)}",
                "count": total_count,  # Total count of detected microplastics
                "counts": counts  # Include counts of each type
            }
        )
    return jsonify({"status": "failure"})

@app.route('/save_results', methods=['POST'])
def save_results():
    data = request.json
    app.logger.info(f"Data received for saving: {data}")  # Log the incoming data
    try:
        # Generate a unique filename based on the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"captured_result_{timestamp}.txt"  # Add an extension

        result = DetectionResult(
            file_name=unique_filename,  # Use the generated unique filename
            station=data["station"],
            beads=data["beads"],
            fragments=data["fragments"],
            fibers=data["fibers"],
            count=data["count"],
            date_created=datetime.utcnow(),  # Store the current date and time
        )
        db.session.add(result)
        db.session.commit()
        app.logger.info("Results saved successfully.")
        return jsonify({"status": "success"})
    except Exception as e:
        app.logger.error(f"Error saving results: {str(e)}")
        db.session.rollback()
        return jsonify({"status": "failure", "error": str(e)}), 500

@app.route('/api/results', methods=['GET'])
@login_required
def get_results():
    results = DetectionResult.query.all()
    return jsonify([result.to_json() for result in results])

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        session_factory = scoped_session(sessionmaker(bind=db.engine))
    app.run(debug=True)