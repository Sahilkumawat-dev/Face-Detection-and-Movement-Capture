from flask import Flask, render_template, jsonify
import cv2
import mediapipe as mp
import time
import os
import threading
from flask import Response
# Support NotesSociety
app = Flask(__name__)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#face detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

#save images on desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_directory = os.path.join(desktop_path, "Saved_Faces")
os.makedirs(save_directory, exist_ok=True)

#movement detection
previous_frame = None
image_counter = 0
movement_threshold = 500
# Adjust value based on your needs
capture_thread = None
cap = None
capture_active = False

def capture_images():
    global previous_frame, image_counter, cap, capture_active
    cap = cv2.VideoCapture(0)
    while capture_active:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        #previous frame
        if previous_frame is None:
            previous_frame = gray_frame
            continue

        # Calculate difference between current frame nd previous frame
        frame_delta = cv2.absdiff(previous_frame, gray_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > movement_threshold:
                # Save the image
                image_counter += 1
                image_path = os.path.join(save_directory, f"person_{image_counter}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Image saved: {image_path}")
                time.sleep(1)  # avoid multiple obj.

        # Draw bounding box around detected faces
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        # Follow NotesSociety
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Update previous frame
        previous_frame = gray_frame

        cv2.imshow("Face Detection and Movement Capture", frame)

        # break loop press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_capture')
def start_capture():
    global capture_active, capture_thread
    if not capture_active:
        capture_active = True
        capture_thread = threading.Thread(target=capture_images, daemon=True)
        capture_thread.start()
        return jsonify({'message': 'Capture started! Move in front of the camera.'})
    return jsonify({'message': 'Capture is already running.'})

@app.route('/stop_capture')
def stop_capture():
    global capture_active
    if capture_active:
        capture_active = False
        return jsonify({'message': 'Capture stopped!'})
    return jsonify({'message': 'Capture is not running.'})

@app.route('/capture_images')
def capture_images_stream():
    def generate():
        global image_counter
        while capture_active:
            yield f"data: {{'counter': {image_counter}}}\n\n"
            time.sleep(1)

    return Response(generate(), content_type='text/event-stream')
# Follow NotesSociety WhatsApp Group
if __name__ == '__main__':
    app.run(debug=True)
