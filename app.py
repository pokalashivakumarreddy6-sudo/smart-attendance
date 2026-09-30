import os
import shutil
import time
import datetime as dt

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

ENROLLED_DIR = "enrolled_faces"
ATTENDANCE_FILE = "attendance.csv"
CONFIDENCE_THRESHOLD = 70  # LBPH distance: LOWER = more confident. Tune this.

st.set_page_config(page_title="Smart Attendance", page_icon="🧑‍💼")
st.title("🧑‍💼 Smart Attendance")
st.caption("Enroll people with your camera, then scan to take attendance. No manual file uploads.")

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
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return cv2.resize(gray[y:y + h, x:x + w], (200, 200))


def _dir_signature(d):
    """A cheap fingerprint of a folder's contents, so the cache below knows when to retrain."""
    if not os.path.isdir(d):
        return ""
    parts = []
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            parts.append(f"{p}:{os.path.getmtime(p)}")
    return "|".join(sorted(parts))


@st.cache_resource(show_spinner="Training on enrolled faces...")
def train_recognizer(signature):
    """
    Scan enrolled_faces/<Name>/*.jpg and fit LBPH.
    'signature' is unused inside but forces Streamlit to retrain whenever the
    folder changes (new enrollment), while reusing the cached model otherwise —
    this cache is shared by every visitor to the app.
    """
    faces, labels, names = [], [], []
    if not os.path.isdir(ENROLLED_DIR):
        return None, []

    for person in sorted(os.listdir(ENROLLED_DIR)):
        person_dir = os.path.join(ENROLLED_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        label_id = len(names)
        found_any = False
        for fname in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
            crop = largest_face_crop(img)
            if crop is None:
                continue
            faces.append(crop)
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
    return bool(already)


# ---------- Sidebar: enroll people ----------
with st.sidebar:
    st.subheader("Enrolled people")
    enrolled_now = sorted(
        p for p in os.listdir(ENROLLED_DIR) if os.path.isdir(os.path.join(ENROLLED_DIR, p))
    ) if os.path.isdir(ENROLLED_DIR) else []
    if enrolled_now:
        for person in enrolled_now:
            n = len(os.listdir(os.path.join(ENROLLED_DIR, person)))
            st.write(f"• {person} ({n} photo(s))")
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
            if largest_face_crop(bgr) is None:
                st.error("No face detected in that photo — try again with better lighting.")
            else:
                person_dir = os.path.join(ENROLLED_DIR, safe_name)
                os.makedirs(person_dir, exist_ok=True)
                cv2.imwrite(os.path.join(person_dir, f"{int(time.time() * 1000)}.jpg"), bgr)
                st.success(f"Enrolled {safe_name}. Saved to disk.")
                st.rerun()

    if enrolled_now:
        st.divider()
        who = st.selectbox("Remove someone", [""] + enrolled_now)
        if who and st.button(f"Remove {who}"):
            shutil.rmtree(os.path.join(ENROLLED_DIR, who), ignore_errors=True)
            st.rerun()

    st.caption("Enrolled people are saved to the app's disk and survive page refreshes. "
               "They reset only if the app itself restarts or redeploys.")

# ---------- Main: live camera recognition ----------
recognizer, known_names = train_recognizer(_dir_signature(ENROLLED_DIR))

if recognizer is None:
    st.info("Nobody is enrolled yet — add people in the sidebar first, then the live camera will appear here.")
else:
    st.subheader("Live attendance camera")
    st.caption("Allow camera access below. Recognized faces are boxed in green and marked present automatically.")

    class FaceRecognitionProcessor(VideoProcessorBase):
        def __init__(self):
            self.recognizer = recognizer
            self.known_names = known_names
            self._last_marked = {}  # name -> unix time, avoids hammering the CSV every frame

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = detect_faces(gray)

            for (x, y, w, h) in boxes:
                crop = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                label_id, distance = self.recognizer.predict(crop)
                if distance <= CONFIDENCE_THRESHOLD:
                    name, color = self.known_names[label_id], (0, 200, 0)
                else:
                    name, color = "Unknown", (0, 0, 255)

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if name != "Unknown":
                    now = time.time()
                    if now - self._last_marked.get(name, 0) > 5:  # re-check at most every 5s per person
                        mark_present(name)
                        self._last_marked[name] = now

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="attendance-live",
        video_processor_factory=FaceRecognitionProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )
    st.caption(
        "If the camera preview doesn't connect, your network may be blocking WebRTC "
        "(common on some corporate/school Wi-Fi). Try a different network or a mobile hotspot."
    )

st.divider()
st.subheader("Attendance log")
if os.path.exists(ATTENDANCE_FILE):
    log = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(log, use_container_width=True)
    st.download_button("Download CSV", log.to_csv(index=False), "attendance.csv")
else:
    st.caption("No attendance marked yet.")
