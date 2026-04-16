import os
import cv2
import torch
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# --- PyTorch Safe Load Fix ---
torch.serialization.add_safe_globals([DetectionModel])

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- AI ACTIVE ON: {device.upper()} ---")

# --- Load Model Once ---
model = YOLO('best.pt')
model.to(device)

# --- CONFIG ---
SAFE_DISTANCE = 30.0  # meters


def stream_processing(source):
    cap = cv2.VideoCapture(source)

    # Set resolution for webcam
    if source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        success, frame = cap.read()
        if not success:
            break

        # --- MODEL INFERENCE ---
        results = model.predict(
            frame,
            conf=0.45,
            imgsz=320,
            verbose=False,
            half=(device == 'cuda')
        )

        annotated_frame = frame.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf_val = float(box.conf[0])

            # --- Distance Estimation ---
            real_h = 1.5
            if 'truck' in cls_name.lower() or 'bus' in cls_name.lower():
                real_h = 3.5
            elif 'person' in cls_name.lower():
                real_h = 1.7

            pixel_h = max((y2 - y1), 1)
            dist = (real_h * (h * 1.2)) / pixel_h

            # --- Draw Only if Unsafe ---
            if dist < SAFE_DISTANCE:
                color = (0, 0, 255)

                label = f"{cls_name.upper()} {conf_val:.2f} | {dist:.1f}m"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # --- Encode Frame ---
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'video_url': '/video_feed/live'})

    file = request.files['video']

    if file.filename == '':
        return jsonify({'video_url': '/video_feed/live'})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    return jsonify({'video_url': f'/video_feed/{file.filename}'})


@app.route('/video_feed/<filename>')
def video_feed(filename):
    source = 0 if filename == 'live' else os.path.join(UPLOAD_FOLDER, filename)

    return Response(
        stream_processing(source),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# --- MAIN ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)