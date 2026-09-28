# Smart Attendance System

Face-recognition attendance with a Streamlit dashboard.

- Detects and encodes faces with `face_recognition` (dlib)
- Matches against enrolled photos in `known_faces/`
- Logs Name / Date / Time to CSV, one entry per person per day
- Live attendance table + CSV download

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Enroll
Add one clear photo per person to `known_faces/` (e.g. `Priya_Sharma.jpg`).
Only enroll people who have consented.
