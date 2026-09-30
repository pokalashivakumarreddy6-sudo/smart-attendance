import os
import datetime as dt

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ATTENDANCE_FILE = "attendance.csv"
CONFIDENCE_THRESHOLD = 70  # LBPH distance: LOWER = more confident. Tune this.

st.set_page_config(page_title="Smart Attendance", page_icon="🧑‍💼")
st.title("🧑‍💼 Smart Attendance")
st.caption("Enroll people with your camera, then scan to take attendance. No files to upload.")

_LOCAL_CASCADE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
_cascade_path = _LOCAL_CASCADE if os.path.exists(_LOCAL_CASCADE) else cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(_cascade_path)
if face_cascade.empty():
    st.error(
        "Could not load the face-detection model file. Make sure "
        "'haarcascade_frontalface_default.xml' is in the same folder as app.py."
    )
    st.stop()


def detect_faces(gray_img):
    """Return list of (x, y, w, h) boxes for faces found in a grayscale image."""
    return face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))


def largest_face_crop(bgr_img):
    """Return the largest detected face as a 200x200 grayscale crop, or None."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    boxes = detect_faces(gray)
    if len(boxes) == 0:
        return None, []
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return cv2.resize(gray[y:y + h, x:x + w], (200, 200)), boxes


if "people" not in st.session_state:
    st.session_state.people = {}  # name -> list of 200x200 grayscale face crops


def train_recognizer():
    """Fit LBPH on everyone currently enrolled this session. None if nobody enrolled."""
    names = list(st.session_state.people.keys())
    if not names:
        return None, []
    faces, labels = [], []
    for label_id, name in enumerate(names):
        for crop in st.session_state.people[name]:
            faces.append(crop)
            labels.append(label_id)
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


# ---------- Sidebar: enroll people ----------
with st.sidebar:
    st.subheader("Enrolled people")
    if st.session_state.people:
        for name in st.session_state.people:
            st.write(f"• {name} ({len(st.session_state.people[name])} photo(s))")
    else:
        st.info("Nobody enrolled yet. Add someone below.")

    st.divider()
    st.subheader("➕ Enroll a new person")
    new_name = st.text_input("Name")
    new_photo = st.camera_input("Take their photo", key="enroll_cam")
    if st.button("Enroll", disabled=not (new_name and new_photo)):
        safe_name = "".join(c for c in new_name.strip() if c.isalnum() or c in (" ", "_", "-")).strip()
        if not safe_name:
            st.error("Please enter a valid name.")
        else:
            img = np.array(Image.open(new_photo).convert("RGB"))
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            crop, _ = largest_face_crop(bgr)
            if crop is None:
                st.error("No face detected in that photo — try again with better lighting.")
            else:
                st.session_state.people.setdefault(safe_name, []).append(crop)
                st.success(f"Enrolled {safe_name}.")
                st.rerun()

    if st.session_state.people:
        st.divider()
        who = st.selectbox("Remove someone", [""] + list(st.session_state.people.keys()))
        if who and st.button(f"Remove {who}"):
            del st.session_state.people[who]
            st.rerun()

    st.caption("⚠️ Enrolled people are remembered only for this browser session — "
               "refreshing the page or reopening the app later starts empty.")

# ---------- Main: scan for attendance ----------
recognizer, known_names = train_recognizer()
scan_photo = st.camera_input("Scan a face to mark attendance")

if scan_photo:
    if recognizer is None:
        st.error("Nobody is enrolled yet — add people in the sidebar first.")
    else:
        img = np.array(Image.open(scan_photo).convert("RGB"))
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
                    name, color = known_names[label_id], (0, 200, 0)
                else:
                    name, color = "Unknown", (0, 0, 255)
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
                st.info("Faces detected, but none matched an enrolled person.")

st.divider()
st.subheader("Attendance log")
if os.path.exists(ATTENDANCE_FILE):
    log = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(log, use_container_width=True)
    st.download_button("Download CSV", log.to_csv(index=False), "attendance.csv")
else:
    st.caption("No attendance marked yet.")
