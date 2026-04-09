# app.py – Complete Criminal Detection System (Final Revision)
import streamlit as st
import cv2
import numpy as np
import os
import shutil
import datetime
import requests
import threading
import time
import base64
import io
import wave
import struct
import math
import sqlite3
from PIL import Image
import streamlit as st
import geocoder
from geopy.geocoders import Nominatim

# 1. Initialize Session State for location so it doesn't refresh constantly
# if 'location' not in st.session_state or st.session_state.location == "Main Entrance":
#     try:
#         # Get coordinates via IP (Fastest for auto-start)
#         g = geocoder.ip('me')
#         if g.latlng:
#             # Reverse geocode to get building/street
#             geolocator = Nominatim(user_agent="criminal_detection_system")
#             location_data = geolocator.reverse(f"{g.latlng[0]}, {g.latlng[1]}", language='en', timeout=5)
            
#             if location_data and 'address' in location_data.raw:
#                 addr = location_data.raw['address']
#                 # Pick the most specific detail available
#                 building = addr.get('building') or addr.get('amenity') or addr.get('office') or addr.get('house_number', '')
#                 road = addr.get('road', '')
#                 city = addr.get('city', addr.get('town', ''))
                
#                 st.session_state.location = f"{building} {road}, {city}".strip(", ")
#             else:
#                 st.session_state.location = f"Lat: {g.latlng[0]}, Lng: {g.latlng[1]}"
#     except Exception:
#         st.session_state.location = "Location Service Timeout"
# -------------------------------------------------------------------
# CONFIGURATION – REPLACE WITH YOUR TELEGRAM CREDENTIALS
# -------------------------------------------------------------------
st.set_page_config(page_title="Criminal Detection System", layout="wide")

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"   # Replace with your bot token
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID"     # Replace with your chat ID

# Paths
HAAR_CASCADE_PATH = 'face_cascade.xml'
DNN_PROTOTXT = 'deploy.prototxt'
DNN_CAFFEMODEL = 'res10_300x300_ssd_iter_140000.caffemodel'
FACE_SAMPLES_DIR = 'face_samples'       # Stores cropped faces per criminal
PROFILE_PICS_DIR = 'profile_pics'       # Stores full profile pictures
DEBUG_DIR = 'debug_faces'               # Debug images with rectangles
DB_PATH = 'criminals.db'

# Face image size for training (must match across registration & recognition)
FACE_TARGET_SIZE = (92, 112)            # width, height

# -------------------------------------------------------------------
# CREATE DIRECTORIES AND DATABASE
# -------------------------------------------------------------------
os.makedirs(FACE_SAMPLES_DIR, exist_ok=True)
os.makedirs(PROFILE_PICS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# SQLite database for criminal records
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS criminals
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              father TEXT,
              mother TEXT,
              gender TEXT,
              dob TEXT,
              blood_group TEXT,
              id_mark TEXT,
              nationality TEXT,
              religion TEXT,
              crimes TEXT)''')
conn.commit()
conn.close()

# -------------------------------------------------------------------
# LOAD FACE DETECTORS
# -------------------------------------------------------------------
# Haar cascade (fallback)
haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
HAAR_AVAILABLE = not haar_cascade.empty()
if HAAR_AVAILABLE:
    print("✓ Haar cascade loaded.")
else:
    print("⚠ Haar cascade not found. Face detection will rely on DNN if available.")

# DNN detector (more accurate)
dnn_net = None
if os.path.isfile(DNN_PROTOTXT) and os.path.isfile(DNN_CAFFEMODEL):
    try:
        dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_CAFFEMODEL)
        print("✓ DNN face detector loaded (more accurate).")
    except Exception as e:
        print(f"✗ Failed to load DNN model: {e}")
else:
    print("ℹ DNN model files not found; using Haar cascade only.")
    print("  For better accuracy, download from:")
    print("  - deploy.prototxt: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt")
    print("  - res10_300x300_ssd_iter_140000.caffemodel: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")

# -------------------------------------------------------------------
# FACE DETECTION FUNCTIONS
# -------------------------------------------------------------------
def detect_faces_haar(gray_frame, scale_factor=1.05, min_neighbors=2, min_size=(20,20)):
    if not HAAR_AVAILABLE:
        return []
    h, w = gray_frame.shape
    # Downscale for speed (optional)
    small = cv2.resize(gray_frame, (w//2, h//2))
    faces = haar_cascade.detectMultiScale(small, scale_factor, min_neighbors, minSize=min_size)
    # Scale back and clip to image boundaries
    valid_faces = []
    for (x, y, fw, fh) in faces:
        x = x*2; y = y*2; fw = fw*2; fh = fh*2
        x = max(0, x); y = max(0, y)
        fw = min(fw, w - x); fh = min(fh, h - y)
        if fw > 0 and fh > 0:
            valid_faces.append((x, y, fw, fh))
    return valid_faces

def detect_faces_dnn(bgr_frame, conf_threshold=0.5):
    if dnn_net is None:
        return []
    h, w = bgr_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            x = max(0, x); y = max(0, y)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 > x and y2 > y:
                faces.append((x, y, x2-x, y2-y))
    return faces

def detect_faces(image):
    """
    Unified face detector: uses DNN if available, otherwise Haar.
    Accepts BGR or grayscale image.
    """
    if dnn_net is not None:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return detect_faces_dnn(image)
    else:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return detect_faces_haar(gray)

# -------------------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------------------
def train_model():
    """
    Train LBPH model from images in FACE_SAMPLES_DIR.
    Returns (model, names_dict).
    """
    model = cv2.face.LBPHFaceRecognizer_create()
    images, labels, names = [], [], {}
    current_id = 0

    for person_name in os.listdir(FACE_SAMPLES_DIR):
        person_path = os.path.join(FACE_SAMPLES_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        names[current_id] = person_name
        for filename in os.listdir(person_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ['.png', '.jpg', '.jpeg', '.pgm']:
                continue
            path = os.path.join(person_path, filename)
            img = cv2.imread(path, 0)   # grayscale
            if img is None:
                continue
            # Resize to target size to ensure uniform shape
            try:
                img_resized = cv2.resize(img, FACE_TARGET_SIZE)
            except:
                continue
            images.append(img_resized)
            labels.append(current_id)
        current_id += 1

    if not images:
        raise ValueError("No training images found. Register a criminal first.")

    images = np.array(images)
    labels = np.array(labels)
    model.train(images, labels)
    print(f"✓ Model trained: {len(set(labels))} subjects, {len(images)} images.")
    return model, names

# -------------------------------------------------------------------
# RECOGNITION FUNCTION (with adjustable threshold)
# -------------------------------------------------------------------

# --- RECOGNITION FUNCTION (With Stricter Thresholding) ---
# ---------------------------------------------------------
def recognize_face(model, frame, gray_frame, face_coords, names, confidence_threshold):
    """
    Annotate frame and return list of verified recognized (name, confidence).
    """
    recognized = []
    recog_names = []

    for (x, y, w, h) in face_coords:
        face = gray_frame[y:y+h, x:x+w]
        try:
            face_resize = cv2.resize(face, FACE_TARGET_SIZE)
        except:
            continue
        
        prediction, confidence = model.predict(face_resize)
        print(f"  Prediction {prediction}: confidence {confidence:.1f}")

        # --- Strict Threshold Check ---
        # LBPH: Lower confidence means a BETTER match.
        # If confidence is > 100, it's almost certainly not a match.
        is_too_weak = confidence > 100 

        if not is_too_weak and confidence < confidence_threshold:
            # 1. Potential match found by algorithm
            potential_label = names.get(prediction, "Unknown").capitalize()
            
            # 2. PRO-LEVEL CHECK: Verify the name exists in the metadata DB
            if potential_label != "Unknown" and verify_registration_exists(potential_label):
                label = potential_label
                if label not in recog_names:
                    recog_names.append(label)
                    recognized.append((label, confidence))
                color = (0, 0, 255)   # red (confirmed match)
            else:
                label = "Unknown"
                color = (0, 255, 0)   # green (unregistered)
        else:
            # Match is too weak to be considered valid
            label = "Unknown"
            color = (0, 255, 0)   # green (unregistered)

        # Draw annotation on the image
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Display "Unknown" or the verified Name
        display_label = label if label == "Unknown" else f"{label} ({confidence:.1f})"
        cv2.putText(frame, display_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, recognized
# def recognize_face(model, frame, gray_frame, face_coords, names, confidence_threshold):
#     """
#     Annotate frame and return list of recognized (name, confidence).
#     """
#     recognized = []
#     recog_names = []

#     for (x, y, w, h) in face_coords:
#         face = gray_frame[y:y+h, x:x+w]
#         try:
#             face_resize = cv2.resize(face, FACE_TARGET_SIZE)
#         except:
#             continue
#         prediction, confidence = model.predict(face_resize)
#         print(f"  Prediction {prediction}: confidence {confidence:.1f} (threshold {confidence_threshold})")

#         if confidence < confidence_threshold:
#             label = names.get(prediction, "Unknown").capitalize()
#             if label not in recog_names:
#                 recog_names.append(label)
#                 recognized.append((label, confidence))
#             color = (0, 0, 255)   # red
#         else:
#             label = "Unknown"
#             color = (0, 255, 0)   # green

#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(frame, f"{label} ({confidence:.1f})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     return frame, recognized

# -------------------------------------------------------------------
# DATABASE HELPERS
# -------------------------------------------------------------------
def insert_criminal(data):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO criminals 
                     (name, father, mother, gender, dob, blood_group,
                      id_mark, nationality, religion, crimes)
                     VALUES (?,?,?,?,?,?,?,?,?,?)''',
                  (data['Name'].lower(),
                   data["Father's Name"].lower(),
                   data["Mother's Name"].lower(),
                   data['Gender'].lower(),
                   data['DOB'],
                   data['Blood Group'].lower(),
                   data['Identification Mark'].lower(),
                   data['Nationality'].lower(),
                   data['Religion'].lower(),
                   data['Crimes Done'].lower()))
        conn.commit()
        row_id = c.lastrowid
        conn.close()
        return row_id
    except Exception as e:
        print(f"DB insert error: {e}")
        return None

def retrieve_criminal(name):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM criminals WHERE name=?", (name.lower(),))
        row = c.fetchone()
        conn.close()
        if row:
            columns = ['id', 'name', 'father', 'mother', 'gender', 'dob',
                       'blood_group', 'id_mark', 'nationality', 'religion', 'crimes']
            data = dict(zip(columns[1:], row[1:]))
            return row[0], data
        else:
            return None, None
    except Exception as e:
        print(f"DB retrieve error: {e}")
        return None, None



# --- DATABASE VERIFICATION HELPER ---
# ------------------------------------
def verify_registration_exists(name):
    """
    Checks if a criminal name actually has metadata records in the DB.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM criminals WHERE name=? LIMIT 1", (name.lower(),))
        row = c.fetchone()
        conn.close()
        return row is not None
    except Exception as e:
        print(f"DB verification error: {e}")
        return False
    
# -------------------------------------------------------------------
# TELEGRAM ALERT
# -------------------------------------------------------------------
# def send_telegram_alert(criminal_name, location):
#     def _send():
#         message = (f"🚨 Criminal Detected!\n"
#                    f"Name: {criminal_name}\n"
#                    f"Location: {location}\n"
#                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
#         data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
#         try:
#             requests.post(url, data=data)
#         except Exception as e:
#             st.error(f"Telegram error: {e}")
#     threading.Thread(target=_send, daemon=True).start()


def send_telegram_alert(criminal_name, location_str):
    def _send():
        # Clean the location string for the URL
        map_query = location_str.replace(" ", "+")
        google_maps_link = f"https://www.google.com/maps/search/?api=1&query={map_query}"
        
        message = (
            f"🚨 **CRIMINAL DETECTED** 🚨\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 **Name:** {criminal_name.upper()}\n"
            f"📍 **Location:** {location_str}\n"
            f"⏰ **Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"🗺️ **View on Map:** {google_maps_link}"
        )
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": message,
            "parse_mode": "Markdown" # Allows bold text in Telegram
        }
        try:
            requests.post(url, data=data)
        except Exception as e:
            st.error(f"Telegram error: {e}")
            
    threading.Thread(target=_send, daemon=True).start()

# -------------------------------------------------------------------
# SIREN SOUND (plays in browser)
# -------------------------------------------------------------------
def play_siren():
    sample_rate = 44100
    duration = 1.5
    freq1 = 1000
    freq2 = 600
    amplitude = 0.5

    samples = []
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        freq = freq1 if (int(t * 4) % 2 == 0) else freq2
        val = amplitude * math.sin(2 * math.pi * freq * t)
        samples.append(val)

    samples_pcm = [int(s * 32767) for s in samples]
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack('<' + 'h'*len(samples_pcm), *samples_pcm))
    wav_bytes = bio.getvalue()
    b64 = base64.b64encode(wav_bytes).decode()
    audio_html = f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# -------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'captured_images' not in st.session_state:
    st.session_state.captured_images = []
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'location' not in st.session_state:
    st.session_state.location = "Eachanari, Coimbatore"
if 'surveillance_active' not in st.session_state:
    st.session_state.surveillance_active = False
if 'already_alerted' not in st.session_state:
    st.session_state.already_alerted = set()
if 'model' not in st.session_state:
    try:
        st.session_state.model, st.session_state.names = train_model()
        st.sidebar.success(f"Model loaded: {len(st.session_state.names)} subjects")
    except ValueError:
        st.session_state.model, st.session_state.names = None, {}
        st.sidebar.info("No criminals registered yet.")

# -------------------------------------------------------------------
# SIDEBAR NAVIGATION AND CONTROLS
# -------------------------------------------------------------------
st.sidebar.title("Navigation")
pages = ["Home", "Register Criminal", "Scan Criminal", "CCTV Surveillance"]
choice = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.page))
st.session_state.page = choice

st.sidebar.subheader("📍 Location")
location_input = st.sidebar.text_input("Current location", value=st.session_state.location)
st.session_state.location = location_input

# Confidence threshold slider – default set to 120 (more forgiving)
st.sidebar.subheader("🎚️ Recognition Threshold")
confidence_threshold = st.sidebar.slider(
    "Confidence (lower = stricter)", 
    min_value=50, max_value=200, value=120, step=5,
    help="Increase this value to make recognition less strict (more likely to match)."
)

def retrain_model():
    with st.spinner("Retraining model..."):
        try:
            st.session_state.model, st.session_state.names = train_model()
            st.sidebar.success(f"Model retrained! {len(st.session_state.names)} subjects now.")
        except Exception as e:
            st.sidebar.error(f"Retrain failed: {e}")

if st.sidebar.button("🔄 Retrain Model"):
    retrain_model()

# Display current known subjects in sidebar for debugging
if st.session_state.names:
    st.sidebar.write("**Known criminals:**")
    for id, name in st.session_state.names.items():
        st.sidebar.write(f"- {name}")

# -------------------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------------------
if st.session_state.page == "Home":
    st.title("🚔 CRIMINAL DETECTION SYSTEM")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 Register Criminal", use_container_width=True):
            st.session_state.page = "Register Criminal"
            st.rerun()
    with col2:
        if st.button("🔍 Scan Criminal", use_container_width=True):
            st.session_state.page = "Scan Criminal"
            st.rerun()
    with col3:
        if st.button("📹 CCTV Surveillance", use_container_width=True):
            st.session_state.page = "CCTV Surveillance"
            st.rerun()
    if os.path.exists("logo.png"):
        st.image("logo.png", width=200)
    st.markdown("---")
    #st.write("Developed by G YESHWANTH GOUD")

# -------------------------------------------------------------------
# REGISTER CRIMINAL PAGE
# -------------------------------------------------------------------
elif st.session_state.page == "Register Criminal":
    st.title("📝 Register Criminal")
    st.markdown("---")

    st.header("1. Provide at least 5 face images")
    tab1, tab2 = st.tabs(["📁 Upload Images", "📸 Capture from Webcam"])

    with tab1:
        uploaded_files = st.file_uploader("Choose images", type=['jpg','jpeg','png'], accept_multiple_files=True)
        if uploaded_files:
            images = []
            for f in uploaded_files:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                images.append(img)
            st.session_state.uploaded_images = images
            st.success(f"{len(images)} images uploaded.")

    with tab2:
        st.write("Click 'Capture' to take a photo. Capture at least 5.")
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            arr = np.frombuffer(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            st.session_state.captured_images.append(img)
            st.success(f"Captured {len(st.session_state.captured_images)} images so far.")

    all_images = st.session_state.uploaded_images + st.session_state.captured_images
    if len(all_images) > 0:
        st.write(f"**Total images ready:** {len(all_images)}")
        cols = st.columns(5)
        for i, img in enumerate(all_images[:5]):
            with cols[i]:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Img {i+1}")

    st.header("2. Criminal Details")
    with st.form("registration_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name *")
            father = st.text_input("Father's Name")
            mother = st.text_input("Mother's Name")
            gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
            # dob = st.date_input("Date of Birth", value=None)
            # Change this:
            # dob = st.date_input("Date of Birth", value=None)

            # To this:
            dob = st.date_input(
                "Date of Birth", 
                value=None, 
                min_value=datetime.date(1900, 1, 1), # Allows dates back to 1900
                max_value=datetime.date.today()      # Prevents future dates
            )
            blood_group = st.selectbox("Blood Group", ["A+","A-","B+","B-","AB+","AB-","O+","O-"])
        with col2:
            id_mark = st.text_input("Identification Mark *")
            nationality = st.text_input("Nationality *")
            religion = st.text_input("Religion *")
            crimes = st.text_area("Crimes Done *")
        profile_index = st.selectbox("Select profile image", list(range(1, len(all_images)+1))) if all_images else 1
        submitted = st.form_submit_button("Register Criminal")

    if submitted:
        if len(all_images) < 5:
            st.error("Please provide at least 5 images.")
        elif not name or not gender or not id_mark or not nationality or not religion or not crimes:
            st.error("Please fill all required fields (*).")
        else:
            data = {
                "Name": name,
                "Father's Name": father,
                "Mother's Name": mother,
                "Gender": gender,
                "DOB": dob.strftime("%Y-%m-%d") if dob else "",
                "Blood Group": blood_group,
                "Identification Mark": id_mark,
                "Nationality": nationality,
                "Religion": religion,
                "Crimes Done": crimes
            }

            # Create temporary folder for this registration
            temp_dir = os.path.join(FACE_SAMPLES_DIR, "temp_criminal")
            os.makedirs(temp_dir, exist_ok=True)

            no_face = []
            for idx, img in enumerate(all_images):
                faces = detect_faces(img)
                # Save debug image with rectangles
                img_debug = img.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"reg_{idx+1}_debug.jpg"), img_debug)

                if len(faces) == 0:
                    no_face.append(idx+1)
                    cv2.imwrite(os.path.join(DEBUG_DIR, f"reg_{idx+1}_original.jpg"), img)
                else:
                    # Convert to grayscale and save the largest face
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Take largest face (by area)
                    largest = max(faces, key=lambda r: r[2]*r[3])
                    x, y, w, h = largest
                    face = gray[y:y+h, x:x+w]
                    if face.shape[0] >= 20 and face.shape[1] >= 20:
                        face_resized = cv2.resize(face, FACE_TARGET_SIZE)
                        cv2.imwrite(os.path.join(temp_dir, f"{idx+1}.pgm"), face_resized)
                    else:
                        no_face.append(idx+1)

            if no_face:
                shutil.rmtree(temp_dir, ignore_errors=True)
                st.error(f"Registration failed! Images without a valid face: {no_face}")
                st.info(f"Check '{DEBUG_DIR}' folder to see why faces weren't detected.")
            else:
                # Insert into database
                row_id = insert_criminal(data)
                if row_id:
                    # Rename temp folder to the criminal's name
                    dest = os.path.join(FACE_SAMPLES_DIR, name)
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.move(temp_dir, dest)
                    # Save profile picture
                    profile_img = all_images[profile_index-1]
                    cv2.imwrite(os.path.join(PROFILE_PICS_DIR, f"criminal_{row_id}.png"), profile_img)
                    st.success("✅ Criminal registered successfully!")
                    # Retrain model
                    retrain_model()
                    # Clear session images
                    st.session_state.uploaded_images = []
                    st.session_state.captured_images = []
                else:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    st.error("Database error. Registration failed.")

# -------------------------------------------------------------------
# SCAN CRIMINAL PAGE
# -------------------------------------------------------------------
elif st.session_state.page == "Scan Criminal":
    st.title("🔍 Scan Criminal")
    st.markdown("---")

    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg','jpeg','png'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input Image", width=400)

        if st.button("🔎 Recognize"):
            if st.session_state.model is None:
                st.error("No model trained. Please register a criminal first.")
            else:
                with st.spinner("Recognizing..."):
                    frame = img.copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detect_faces(frame)
                    if len(faces) == 0:
                        st.info("No face detected in the image.")
                    else:
                        annotated_frame, recognized = recognize_face(
                            st.session_state.model, frame, gray, faces,
                            st.session_state.names, confidence_threshold
                        )
                        # if recognized:
                        #     play_siren()
                        #     for crim in recognized:
                        #         send_telegram_alert(crim[0], st.session_state.location)

                        if recognized:
                            play_siren()
                            for name, conf in recognized:
                                # Pass the verified name and the automatic location
                                send_telegram_alert(name, st.session_state.location)


                        col1, col2 = st.columns([2, 1]) # 2 parts image, 1 part details
                        
                        with col1:
                            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                                     caption="Annotated Analysis", use_container_width=True)
                        
                        with col2:
                            if recognized:
                                st.subheader("📋 Criminal Details")
                                for name, conf in recognized:
                                    # Use the retrieval function from your app.py
                                    db_id, data = retrieve_criminal(name)
                                    
                                    if db_id:
                                        # Show Profile Picture automatically
                                        prof_path = os.path.join(PROFILE_PICS_DIR, f"criminal_{db_id}.png")
                                        if os.path.exists(prof_path):
                                            st.image(prof_path, width=150, caption="Database Photo")

                                        # Display critical fields from your DB
                                        st.info(f"📍 **Last Seen At:** {st.session_state.location}") 

                                        st.markdown(f"**Name:** {data.get('name', 'N/A').title()}")
                                        st.markdown(f"**Gender:** {data.get('gender', 'N/A')}")
                                        st.markdown(f"**Nationality:** {data.get('nationality', 'N/A')}")
                                        st.error(f"**Crimes Done:** {data.get('crimes', 'N/A')}")
                                        
                                        # Secondary details in expander
                                        with st.expander("View Full Background"):
                                            st.write(f"**Father:** {data.get('father', 'N/A')}")
                                            st.write(f"**ID Mark:** {data.get('id_mark', 'N/A')}")
                                            st.write(f"**Blood Group:** {data.get('blood_group', 'N/A')}")
                                            st.write(f"**DOB:** {data.get('dob', 'N/A')}")
                                        st.markdown("---")
                            else:
                                st.info("No known criminals recognized.")

                        # col1, col2 = st.columns(2)
                        # with col1:
                        #     st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Annotated")
                        # with col2:
                        #     if recognized:
                        #         st.subheader("Detected Criminals")
                        #         for name, conf in recognized:
                        #             if st.button(f"👤 {name}", key=f"scan_{name}"):
                        #                 id, data = retrieve_criminal(name)
                        #                 if id:
                        #                     with st.expander(f"Profile: {name}", expanded=True):
                        #                         col_a, col_b = st.columns([1,2])
                        #                         with col_a:
                        #                             prof_path = os.path.join(PROFILE_PICS_DIR, f"criminal_{id}.png")
                        #                             if os.path.exists(prof_path):
                        #                                 st.image(prof_path, width=200)
                        #                         with col_b:
                        #                             for k,v in data.items():
                        #                                 st.write(f"**{k}:** {v}")
                        #     else:
                        #         st.info("No known criminals recognized.")

# -------------------------------------------------------------------
# CCTV SURVEILLANCE PAGE
# -------------------------------------------------------------------
elif st.session_state.page == "CCTV Surveillance":
    st.title("📹 CCTV Surveillance")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Start Surveillance"):
            st.session_state.surveillance_active = True
            st.session_state.already_alerted = set()
    with col2:
        if st.button("⏹️ Stop Surveillance"):
            st.session_state.surveillance_active = False
            st.rerun()
    with col3:
        st.write(f"**📍 Location:** {st.session_state.location}")

    video_placeholder = st.empty()
    detected_placeholder = st.empty()

    if st.session_state.surveillance_active:
        if st.session_state.model is None:
            st.error("No model trained. Please register a criminal first.")
            st.session_state.surveillance_active = False
        else:
            st.warning("Surveillance is running. Click 'Stop' to end.")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam.")
                st.session_state.surveillance_active = False
            else:
                while st.session_state.surveillance_active:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detect_faces(frame)

                    if len(faces) > 0:
                        annotated_frame, recognized = recognize_face(
                            st.session_state.model, frame.copy(), gray, faces,
                            st.session_state.names, confidence_threshold
                        )
                        recog_names = [r[0] for r in recognized]

                        new_names = set(recog_names) - st.session_state.already_alerted
                        if new_names:
                            play_siren()
                            for name in new_names:
                                send_telegram_alert(name, st.session_state.location)
                                st.session_state.already_alerted.add(name)
                                
                    else:
                        annotated_frame = frame
                        recog_names = []

                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)

                    # ... inside the surveillance loop ...
                    if recog_names:
                        with detected_placeholder.container():
                            st.markdown("### 🚨 Live Identification")
                            for name in set(recog_names):
                                db_id, data = retrieve_criminal(name)
                                if db_id:
                                    # Creates a clean card for each person detected live
                                    with st.container(border=True):
                                        c1, c2 = st.columns([1, 2])
                                        with c1:
                                            prof_path = os.path.join(PROFILE_PICS_DIR, f"criminal_{db_id}.png")
                                            if os.path.exists(prof_path):
                                                st.image(prof_path, use_container_width=True)
                                        with c2:
                                            st.markdown(f"**Target:** {name.upper()}")
                                            st.warning(f"📍 **Last Seen At:** {st.session_state.location}")
                                            st.error(f"**Crimes:** {data.get('crimes', 'N/A')}")
                                            st.info(f"**Mark:** {data.get('id_mark', 'N/A')}")
                                            
                    # if recog_names:
                    #     with detected_placeholder.container():
                    #         st.subheader("Detected Criminals")
                    #         for name in set(recog_names):
                    #             if st.button(f"👤 {name}", key=f"surv_{name}_{time.time()}"):
                    #                 id, data = retrieve_criminal(name)
                    #                 if id:
                    #                     with st.expander(f"Profile: {name}", expanded=True):
                    #                         col_a, col_b = st.columns([1,2])
                    #                         with col_a:
                    #                             prof_path = os.path.join(PROFILE_PICS_DIR, f"criminal_{id}.png")
                    #                             if os.path.exists(prof_path):
                    #                                 st.image(prof_path, width=200)
                    #                         with col_b:
                    #                             for k,v in data.items():
                    #                                 st.write(f"**{k}:** {v}")
                    else:
                        detected_placeholder.empty()

                    time.sleep(0.03)

                cap.release()
                st.success("Surveillance stopped.")
    else:
        st.info("Click 'Start Surveillance' to begin.")

# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.caption("Criminal Detection System © 2026")