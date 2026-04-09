# register.py
import cv2
import os
from facerec import detect_faces

FACE_TARGET_SIZE = (92, 112)   # width, height

def registerCriminal(img, dest_path, img_id):
    """
    Save a face image to the destination folder after verifying it contains a face.
    img: can be color (BGR) or grayscale.
    Returns img_id if successful, None if no valid face detected.
    """
    # Convert to grayscale if it's color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Detect faces
    faces = detect_faces(gray)   # returns list of (x, y, w, h)
    if len(faces) == 0:
        return None

    # Choose the largest face (by area)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    if w <= 0 or h <= 0:
        return None

    # Extract the face region
    face = gray[y:y+h, x:x+w]

    # Minimum size check
    if face.shape[0] < 20 or face.shape[1] < 20:
        return None

    # Resize to standard dimensions
    try:
        face_resized = cv2.resize(face, FACE_TARGET_SIZE)
    except cv2.error:
        return None

    # Save the face image
    filename = os.path.join(dest_path, f"{img_id}.pgm")
    cv2.imwrite(filename, face_resized)
    return img_id