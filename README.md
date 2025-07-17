# Emotion-Based Music Player 🎶😄

This project is a smart music player that detects your emotion via webcam and plays music that suits your mood. It uses a deep learning model for facial emotion recognition and categorizes songs into mood-based folders.

---

## 🗂️ Project Structure

```
EmotionMusicPlayer/
├── main.py                     # Main script to run the app
├── emotion_log.csv             # Logs of detected emotions with timestamps
├── requirements.txt            # List of Python dependencies
├── models/
│   └── emotion_model.h5        # Pre-trained emotion detection model
├── music/
│   ├── angry/
│   │   └── last_action_hero.mp3
│   ├── happy/
│   │   └── be_happy.mp3
│   ├── neutral/
│   │   └── neutral_to_second.mp3
│   └── sad/
│       └── sad_ringtone.mp3
```

---

## 💻 How to Run the Project

1. **Install Python 3.7+**

2. **Install Dependencies**

   Run this in the terminal:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```

4. **Use Your Webcam**
   - The app will detect your face and classify the emotion.
   - A song matching your mood will play from the `music/` folder.

---

## 😎 Supported Emotions

- Happy
- Sad
- Angry
- Neutral

---

## 📋 Log File

- The `emotion_log.csv` file records each detected emotion with a timestamp.

---

## 🤖 Model Info

- The file `emotion_model.h5` is a trained Keras model for real-time emotion recognition using facial landmarks and CNN.

---

## 📌 Notes

- Music is selected randomly from each emotion folder.
- Make sure your webcam is connected and working.
- You can add more `.mp3` files in respective emotion folders to personalize the experience.

---

## 📄 License

This project is for educational and personal use. You may modify and distribute it under the terms of the MIT license.

---

Built with 💙 using Python + OpenCV + Deep Learning.
# Emotion-Music-Player
