import cv2
import numpy as np
import os
import random
from keras.models import load_model
from datetime import datetime
import csv
from pygame import mixer


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MUSIC_DIR = 'music'
MODEL_PATH = 'models/emotion_model.h5'
LOG_FILE = 'emotion_log.csv'

#Load Emotion Model 

model = load_model(MODEL_PATH, compile=False) 
mixer.init()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def log_emotion(emotion):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion])

# Play Music
def play_music(emotion):
    folder = os.path.join(MUSIC_DIR, emotion.lower())
    if not os.path.exists(folder):
        print(f"[WARNING] No music found for '{emotion}' emotion.")
        return
    songs = [f for f in os.listdir(folder) if f.endswith('.mp3')]
    if songs:
        song = random.choice(songs)
        song_path = os.path.join(folder, song)
        try:
            mixer.music.load(song_path)
            mixer.music.play()
            print(f"[INFO] Playing '{song}' for emotion: {emotion}")
        except Exception as e:
            print(f"[ERROR] Could not play music: {e}")
    else:
        print(f"[WARNING] No songs found in folder: {folder}")

#Emotion Detection

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        try:
            roi = cv2.resize(roi, (64, 64)) / 255.0  
        except:
            continue
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)[0]
        emotion = EMOTION_LABELS[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

        return emotion
    return None



def main():
    print("🔊 Starting Emotion Music Player...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    last_emotion = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        emotion = detect_emotion(frame)
        if emotion and emotion != last_emotion:
            log_emotion(emotion)
            play_music(emotion)
            last_emotion = emotion

        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mixer.quit()
    print("🎵 Emotion Music Player stopped.")

if __name__ == "__main__":
    main()
