import os
import cv2
import torch
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# 1. PyTorch 2.6 Security Fix
torch.serialization.add_safe_globals([DetectionModel])

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 2. Hardware Acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- AI ACTIVE ON: {device.upper()} ---")
# Load model once
model = YOLO('best.pt').to(device)

# --- CONFIGURATION ---
SAFE_DISTANCE = 30.0  # Meters

def stream_processing(source):
    cap = cv2.VideoCapture(source)
    
    # Optional: If using live cam, force a lower resolution to keep FPS high
    if source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # --- EVERY FRAME PROCESSING ---
        # We removed the 'if frame_count % SKIP_FRAMES' block.
        
        # Optimize: imgsz=320 and half-precision are vital for every-frame speed
        results = model.predict(
            frame, 
            conf=0.45, 
            imgsz=320, 
            verbose=False, 
            half=(device == 'cuda') # Use FP16 on GPU for speed
        )
        
        annotated_frame = frame.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = model.names[int(box.cls[0])]
            conf_val = float(box.conf[0])
            
            # Distance Logic
            real_h = 1.5
            if any(k in cls_name.lower() for k in ['truck', 'bus']):
                real_h = 3.5
            elif 'person' in cls_name.lower():
                real_h = 1.7
            
            pixel_h = y2 - y1
            dist = (real_h * (h * 1.2)) / (pixel_h + 1e-6)

            # Red Box Logic
            if dist < SAFE_DISTANCE:
                color = (0, 0, 255) 
                label = f"{cls_name.upper()} {conf_val:.2f} | {dist:.1f}m"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

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

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    return jsonify({'video_url': f'/video_feed/{filename}'})

@app.route('/video_feed/<filename>')
def video_feed(filename):
    source = 0 if filename == 'live' else os.path.join(UPLOAD_FOLDER, filename)
    return Response(stream_processing(source), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)