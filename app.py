"""
Smart Attendance — face recognition with OpenCV only (no dlib, no compiling).

How it works:
1. Known faces live in known_faces/<PersonName>/*.jpg (one or more photos per person).
2. On startup we detect faces in those photos (Haar cascade) and train an
   LBPH recognizer (cv2.face.LBPHFaceRecognizer) to tell people apart.
3. Upload a new photo -> we detect faces in it and ask the recognizer who each one is.
4. Recognized people get marked present in attendance.csv (once per day).
"""
import os
import datetime as dt

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

KNOWN_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
CONFIDENCE_THRESHOLD = 70  # LBPH distance: LOWER = more confident. Tune this.

st.set_page_config(page_title="Smart Attendance", page_icon="🧑‍💼")
st.title("🧑‍💼 Smart Attendance")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_faces(gray_img):
    """Return list of (x, y, w, h) boxes for faces found in a grayscale image."""
    return face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))


@st.cache_resource(show_spinner="Training on known faces...")
def train_recognizer():
    """Read known_faces/<Name>/*.jpg, detect the face crop in each, and fit LBPH."""
    faces, labels, names = [], [], []
    if not os.path.isdir(KNOWN_DIR):
        return None, []

    for person in sorted(os.listdir(KNOWN_DIR)):
        person_dir = os.path.join(KNOWN_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        label_id = len(names)
        found_any = False
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = detect_faces(gray)
            if len(boxes) == 0:
                continue
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])  # largest face in photo
            faces.append(cv2.resize(gray[y:y + h, x:x + w], (200, 200)))
            labels.append(label_id)
            found_any = True
        if found_any:
            names.append(person)

    if not faces:
        return None, []

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    return recognizer, names


def mark_present(name):
    today = dt.date.today().isoformat()
    now = dt.datetime.now().strftime("%H:%M:%S")
    cols = ["Name", "Date", "Time"]
    df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame(columns=cols)
    already = ((df["Name"] == name) & (df["Date"] == today)).any()
    if not already:
        df = pd.concat([df, pd.DataFrame([[name, today, now]], columns=cols)], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
    return already


recognizer, known_names = train_recognizer()

with st.sidebar:
    st.subheader("Known people")
    if known_names:
        st.write(", ".join(known_names))
    else:
        st.warning(f"No known faces found. Add photos under `{KNOWN_DIR}/<PersonName>/`.")
    st.caption("One clear, front-facing photo per person is enough. More photos improve accuracy.")

uploaded = st.file_uploader("Upload a photo to take attendance", type=["jpg", "jpeg", "png"])

if uploaded and recognizer is not None:
    img = np.array(Image.open(uploaded).convert("RGB"))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    boxes = detect_faces(gray)

    if len(boxes) == 0:
        st.warning("No faces detected in this photo.")
    else:
        marked = []
        for (x, y, w, h) in boxes:
            face_crop = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            label_id, distance = recognizer.predict(face_crop)
            if distance <= CONFIDENCE_THRESHOLD:
                name = known_names[label_id]
                color = (0, 200, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)
            cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
            cv2.putText(bgr, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if name != "Unknown":
                already = mark_present(name)
                marked.append(f"{name} ({'already marked today' if already else 'marked present'})")

        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Detected faces")
        if marked:
            for m in marked:
                st.success(m)
        else:
            st.info("Faces detected, but none matched a known person.")
elif uploaded:
    st.error("No known faces to compare against yet — add photos to known_faces/ first.")

st.divider()
st.subheader("Attendance log")
if os.path.exists(ATTENDANCE_FILE):
    log = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(log, use_container_width=True)
    st.download_button("Download CSV", log.to_csv(index=False), "attendance.csv")
else:
    st.caption("No attendance marked yet.")
