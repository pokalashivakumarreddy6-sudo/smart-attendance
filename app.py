"""Smart Attendance System (face recognition).

Setup:
  pip install streamlit face_recognition pandas pillow numpy
  (face_recognition needs dlib; on Colab/Linux: pip install cmake dlib)

Enroll people: put one clear photo per person in known_faces/
  named after them, e.g. known_faces/Priya_Sharma.jpg

Run: streamlit run attendance_app.py
"""
import os
from datetime import datetime

import face_recognition
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

KNOWN_DIR = "known_faces"
LOG_FILE = "attendance.csv"
TOLERANCE = 0.5  # lower = stricter matching

st.set_page_config(page_title="Smart Attendance", page_icon="✅")
st.title("✅ Smart Attendance")


@st.cache_resource
def load_known():
    names, encodings = [], []
    os.makedirs(KNOWN_DIR, exist_ok=True)
    for f in os.listdir(KNOWN_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = face_recognition.load_image_file(os.path.join(KNOWN_DIR, f))
            enc = face_recognition.face_encodings(img)
            if enc:
                names.append(os.path.splitext(f)[0].replace("_", " "))
                encodings.append(enc[0])
    return names, encodings


def mark(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(LOG_FILE) if os.path.exists(LOG_FILE) else pd.DataFrame(columns=["Name", "Date", "Time"])
    if ((df["Name"] == name) & (df["Date"] == today)).any():
        return False
    df.loc[len(df)] = [name, today, now]
    df.to_csv(LOG_FILE, index=False)
    return True


names, encodings = load_known()
st.caption(f"{len(names)} people enrolled")

shot = st.camera_input("Look at the camera")
if shot:
    img = np.array(Image.open(shot).convert("RGB"))
    locs = face_recognition.face_locations(img)
    encs = face_recognition.face_encodings(img, locs)
    if not encs:
        st.warning("No face detected. Try better lighting.")
    for enc in encs:
        if not encodings:
            st.error("No enrolled faces found in known_faces/")
            break
        dist = face_recognition.face_distance(encodings, enc)
        best = int(np.argmin(dist))
        if dist[best] <= TOLERANCE:
            name = names[best]
            if mark(name):
                st.success(f"Welcome, {name}! Attendance marked.")
            else:
                st.info(f"{name}, you're already marked present today.")
        else:
            st.error("Face not recognized.")

st.divider()
st.subheader("Today's attendance")
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    st.dataframe(df[df["Date"] == today], use_container_width=True)
    st.download_button("Download full log (CSV)", df.to_csv(index=False), "attendance.csv")
else:
    st.write("No records yet.")
