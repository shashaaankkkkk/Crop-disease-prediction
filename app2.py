import os
import cv2
import sqlite3
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, render_template, request

app = Flask(__name__)

# Initialize database
def init_db():
    conn = sqlite3.connect('photos.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS photos
                 (id INTEGER PRIMARY KEY, filename TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

# Check if the database exists and initialize it if not
def check_and_initialize_db():
    if not os.path.exists('photos.db'):
        init_db()

# Capture a single photo
def capture_photo(photo_id, frame):
    image_folder = "captured_images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Generate unique filename based on photo_id and current timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"photo_{photo_id}_{timestamp}.jpg"
    filepath = os.path.join(image_folder, filename)

    # Save the frame as an image
    cv2.imwrite(filepath, frame)

    # Ensure the database is initialized before saving
    check_and_initialize_db()

    # Save the photo details to the database
    conn = sqlite3.connect('photos.db', check_same_thread=False)
    c = conn.cursor()
    timestamp_db = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO photos (filename, timestamp) VALUES (?, ?)", (filename, timestamp_db))
    conn.commit()
    conn.close()

    return filename

# API to capture a single photo from the video stream
@app.route('/capture', methods=['POST'])
def capture_from_stream():
    # Ensure the database is initialized before capturing
    check_and_initialize_db()

    # URL of the video stream
    video_url = "http://192.168.232.102:5000/video_feed"

    # Open video stream
    cap = cv2.VideoCapture(video_url)

    # Check if the video stream is open
    if not cap.isOpened():
        return jsonify({'error': 'Failed to open video stream'}), 500

    # Read a single frame from the stream
    ret, frame = cap.read()
    cap.release()  # Release the video capture after reading one frame

    # Check if frame was successfully captured
    if not ret:
        return jsonify({'error': 'Failed to capture frame'}), 500

    # Generate a photo ID (could be auto-increment or another logic)
    photo_id = int(datetime.now().timestamp())  # Simple unique ID generation based on timestamp

    # Capture and save the photo
    filename = capture_photo(photo_id, frame)

    return jsonify({'message': 'Photo captured successfully', 'filename': filename}), 200

# API to view photo by ID
@app.route('/api/photos/<int:id>', methods=['GET'])
def get_photo(id):
    # Ensure the database is initialized before fetching photos
    check_and_initialize_db()

    conn = sqlite3.connect('photos.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT filename FROM photos WHERE id=?", (id,))
    photo = c.fetchone()
    conn.close()

    if photo:
        image_folder = "captured_images"
        return send_from_directory(image_folder, photo[0])
    else:
        return jsonify({'error': 'Photo not found'}), 404

# API to get a list of all photos
@app.route('/api/photos', methods=['GET'])
def list_photos():
    # Ensure the database is initialized before listing photos
    check_and_initialize_db()

    conn = sqlite3.connect('photos.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, filename FROM photos")
    photos = c.fetchall()
    conn.close()

    return jsonify({'photos': [{'id': photo[0], 'filename': photo[1]} for photo in photos]})

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream video feed
def generate_frames():
    video_url = "http://192.168.232.102:5000/video_feed"
    cap = cv2.VideoCapture(video_url)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # No rotation applied here

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return app.response_class(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Initialize the database on app startup if it doesn't exist
    check_and_initialize_db()
    app.run(debug=True)
