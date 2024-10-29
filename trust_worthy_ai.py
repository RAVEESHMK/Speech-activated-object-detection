import numpy as np
import cv2
import os
import subprocess
import threading
from gtts import gTTS

ffmpeg_path = "C:/FFpeg"
LABELS = open("D:/raveesh/yolo-master/yolo-coco/coco.names").read().strip().split("\n")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

cap = cv2.VideoCapture(0)
audio_playing = False

def play_audio(description):
    global audio_playing
    audio_playing = True
    tts = gTTS(description, lang='en')
    tts.save("tts.mp3")
    subprocess.call([os.path.join(ffmpeg_path, "ffplay.exe"), "-nodisp", "-autoexit", "tts.mp3"])
    audio_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes, confidences, classIDs = [], [], []
    current_objects = set()

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, width, height) = boxes[i]
            label = f"{LABELS[classIDs[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            position = ""
            if centerX <= frame.shape[1] / 3:
                position += "left "
            elif centerX <= (frame.shape[1] / 3 * 2):
                position += "center "
            else:
                position += "right "

            if centerY <= frame.shape[0] / 3:
                position += "top "
            elif centerY <= (frame.shape[0] / 3 * 2):
                position += "middle "
            else:
                position += "bottom "

            current_objects.add(f"{position.strip()} {LABELS[classIDs[i]]}")

    # Only play audio if not currently playing and if we have new detections
    if not audio_playing and current_objects:
        description = ', '.join(current_objects)
        threading.Thread(target=play_audio, args=(description,)).start()

    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if os.path.exists("tts.mp3"):
    os.remove("tts.mp3")
