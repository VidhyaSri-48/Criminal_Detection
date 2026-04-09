# facerec.py
import cv2
import numpy as np
import os
import warnings

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
size = 2                         # downscaling factor for Haar
HAAR_CASCADE_PATH = 'face_cascade.xml'
DNN_PROTOTXT = 'deploy.prototxt'
DNN_CAFFEMODEL = 'res10_300x300_ssd_iter_140000.caffemodel'
TARGET_SIZE = (92, 112)          # width, height for training images

# -------------------------------------------------------------------
# Load Haar cascade
# -------------------------------------------------------------------
haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if haar_cascade.empty():
    warnings.warn(f"Haar cascade file '{HAAR_CASCADE_PATH}' not found or empty. Face detection will fail unless DNN is used.")
    HAAR_AVAILABLE = False
else:
    print("✓ Haar cascade loaded.")
    HAAR_AVAILABLE = True

# -------------------------------------------------------------------
# Load DNN face detector (if model files exist)
# -------------------------------------------------------------------
dnn_net = None
if os.path.isfile(DNN_PROTOTXT) and os.path.isfile(DNN_CAFFEMODEL):
    try:
        dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_CAFFEMODEL)
        print("✓ DNN face detector loaded (more accurate).")
    except Exception as e:
        print(f"✗ Failed to load DNN model: {e}")
        dnn_net = None
else:
    print("ℹ DNN model files not found; using Haar cascade only.")
    print("  For better accuracy, download from:")
    print("  - deploy.prototxt: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt")
    print("  - res10_300x300_ssd_iter_140000.caffemodel: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")

# -------------------------------------------------------------------
# Training function
# -------------------------------------------------------------------
def train_model():
    """
    Train an LBPH face recognizer using images in 'face_samples/'.
    All images are resized to TARGET_SIZE to ensure uniform shape.
    Returns (model, names_dict).
    """
    model = cv2.face.LBPHFaceRecognizer_create()
    fn_dir = 'face_samples'
    print('Training...')

    images, labels, names = [], [], {}
    current_id = 0

    for subdir, dirs, files in os.walk(fn_dir):
        for person_name in dirs:
            names[current_id] = person_name
            subject_path = os.path.join(fn_dir, person_name)
            for filename in os.listdir(subject_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.pgm']:
                    continue
                path = os.path.join(subject_path, filename)
                img = cv2.imread(path, 0)          # read as grayscale
                if img is None:
                    print(f"  Warning: Could not read {path}, skipping.")
                    continue
                # Resize to target size to ensure all images have same dimensions
                try:
                    img_resized = cv2.resize(img, TARGET_SIZE)
                except cv2.error as e:
                    print(f"  Warning: Could not resize {path}: {e}, skipping.")
                    continue
                images.append(img_resized)
                labels.append(current_id)
            current_id += 1

    if not images:
        raise ValueError("No training images found. Check the 'face_samples' directory.")

    images = np.array(images)
    labels = np.array(labels)
    model.train(images, labels)
    print(f"✓ Training complete. {len(set(labels))} subjects, {len(images)} images.")
    return model, names

# -------------------------------------------------------------------
# Face detection functions
# -------------------------------------------------------------------
def detect_faces_haar(gray_frame):
    """
    Detect faces using Haar cascade with sensitive parameters.
    gray_frame: grayscale image (numpy array)
    Returns list of (x, y, w, h) in original image coordinates,
    clipped to image boundaries.
    """
    if not HAAR_AVAILABLE:
        return []
    global size
    h, w = gray_frame.shape
    small = cv2.resize(gray_frame, (int(w/size), int(h/size)))
    faces = haar_cascade.detectMultiScale(
        small,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(20, 20)
    )
    full_faces = []
    for (x, y, fw, fh) in faces:
        x = x * size
        y = y * size
        fw = fw * size
        fh = fh * size
        x = max(0, x)
        y = max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)
        if fw > 0 and fh > 0:
            full_faces.append((x, y, fw, fh))
    if full_faces:
        print(f"  Haar found {len(full_faces)} face(s)")
    return full_faces

def detect_faces_dnn(bgr_frame):
    """
    Detect faces using DNN (most accurate).
    bgr_frame: color image (BGR order)
    Returns list of (x, y, w, h), clipped to image boundaries.
    """
    if dnn_net is None:
        return []
    h, w = bgr_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    faces = []
    confidences = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:   # confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 > x and y2 > y:
                faces.append((x, y, x2-x, y2-y))
                confidences.append(confidence)
    if confidences:
        print(f"  DNN found {len(faces)} face(s) with max confidence {max(confidences):.2f}")
    return faces

def detect_faces(image):
    """
    Unified face detection: uses DNN if available, otherwise Haar.
    Accepts either color (BGR) or grayscale image.
    Returns list of (x, y, w, h) – always valid and non‑empty.
    """
    print("detect_faces called with image shape:", image.shape)
    if dnn_net is not None:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            print("  Converted grayscale to BGR for DNN")
        faces = detect_faces_dnn(image)
    else:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        faces = detect_faces_haar(gray)
    return faces

# -------------------------------------------------------------------
# Recognition function with adjustable threshold
# -------------------------------------------------------------------
def recognize_face(model, frame, gray_frame, face_coords, names, confidence_threshold=95):
    """
    Annotate frame with recognized names and return list of detected criminals.
    confidence_threshold: maximum allowed confidence (lower is better). 
                         Increase this value to make recognition less strict.
    """
    img_width, img_height = TARGET_SIZE
    recognized = []
    recog_names = []

    for (x, y, w, h) in face_coords:
        if w <= 0 or h <= 0:
            continue
        face = gray_frame[y:y+h, x:x+w]
        try:
            face_resize = cv2.resize(face, (img_width, img_height))
        except cv2.error:
            continue

        prediction, confidence = model.predict(face_resize)
        print(f"  Confidence for prediction {prediction}: {confidence:.1f} (threshold={confidence_threshold})")

        if confidence < confidence_threshold:          # known criminal
            label = names.get(prediction, "Unknown").capitalize()
            if label not in recog_names:
                recog_names.append(label)
                recognized.append((label, confidence))
            color = (0, 0, 255)      # red
        else:                         # unknown
            label = "Unknown"
            color = (0, 255, 0)       # green

        # Draw rectangle and label with confidence
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.1f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, recognized