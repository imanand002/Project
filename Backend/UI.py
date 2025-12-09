# import tkinter as tk
# import cv2
# import numpy as np
# from keras.models import model_from_json
# from keras.preprocessing import image
# import keras.utils as image
# from PIL import Image, ImageTk

# # Load the pre-trained model
# model = model_from_json(open("model_fer.json", "r").read())
# model.load_weights('model_fer.h5')

# # Initialize the face cascade classifier
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# # Function to process video frames and detect emotions
# def detect_emotions():
#     ret, frame = cap.read()
#     if not ret:
#         return

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         img_pixels = image.img_to_array(roi_gray)
#         img_pixels = np.expand_dims(img_pixels, axis=0)
#         img_pixels /= 255.0

#         predictions = model.predict(img_pixels)
#         max_index = np.argmax(predictions[0])
#         emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#         predicted_emotion = emotions[max_index]

#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(frame)
#     # img = Image.fromarray((frame * 255).astype('uint8'))
#     imgtk = ImageTk.PhotoImage(image=img)
#     video_label.imgtk = imgtk
#     video_label.configure(image=imgtk)

#     video_label.after(10, detect_emotions)

# # Function to close the application
# def close_application():
#     window.destroy()

# # Create the main window
# window = tk.Tk()
# window.title("Real-Time Emotion Detection")
# window.geometry("800x600")

# # Create a label to display the video stream
# video_label = tk.Label(window)
# video_label.pack()

# # Create a button to close the application
# close_button = tk.Button(window, text="Close", command=close_application)
# close_button.place(relx=.5, rely=.8)
# close_button.pack()

# # Open the video capture
# cap = cv2.VideoCapture(0)

# # Start the emotion detection process
# detect_emotions()

# # Run the Tkinter event loop
# window.mainloop()

# # Release the video capture and destroy the window
# cap.release()
# cv2.destroyAllWindows()
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QComboBox, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from keras.models import model_from_json
from keras.preprocessing import image
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class EmotionDetectionThread(QThread):
    frame_processed = pyqtSignal(object, str, float, dict)
    
    def __init__(self, model, face_cascade):
        super().__init__()
        self.model = model
        self.face_cascade = face_cascade
        self.cap = cv2.VideoCapture(0)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.running = True
    
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            annotated_frame = frame.copy()
            dominant_emotion = "N/A"
            confidence = 0.0
            emotion_scores = {}
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    img_pixels = image.img_to_array(roi_gray)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255.0
                    
                    predictions = self.model.predict(img_pixels, verbose=0)[0]
                    max_index = np.argmax(predictions)
                    dominant_emotion = self.emotions[max_index]
                    confidence = float(predictions[max_index])
                    
                    # Create emotion scores dict
                    emotion_scores = {self.emotions[i]: float(predictions[i]) for i in range(len(self.emotions))}
                    
                    # Draw bounding box with emotion
                    color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(annotated_frame, f"{dominant_emotion} ({confidence:.2f})", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            self.frame_processed.emit(annotated_frame, dominant_emotion, confidence, emotion_scores)
    
    def stop(self):
        self.running = False
        self.cap.release()

class EmotionDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load model
        self.model = model_from_json(open("model_fer.json", "r").read())
        self.model.load_weights('model_fer.h5')
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        self.initUI()
        self.emotion_history = deque(maxlen=100)
        
    def initUI(self):
        self.setWindowTitle('🎭 Real-Time Emotion Detection')
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QLabel { color: #ffffff; }
            QPushButton { 
                background-color: #0066cc; 
                color: white; 
                border: none; 
                padding: 10px; 
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0052a3; }
        """)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        
        # Left side - Video display
        left_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000; border: 2px solid #0066cc;")
        left_layout.addWidget(self.video_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton('▶ Start Detection')
        self.stop_btn = QPushButton('⏹ Stop Detection')
        self.stop_btn.setEnabled(False)
        self.capture_btn = QPushButton('📸 Capture Screenshot')
        
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.capture_btn.clicked.connect(self.capture_screenshot)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.capture_btn)
        left_layout.addLayout(button_layout)
        
        # Right side - Stats and graphs
        right_layout = QVBoxLayout()
        
        # Current emotion display
        self.emotion_label = QLabel('Emotion: Waiting...')
        self.emotion_label.setFont(QFont('Arial', 24, QFont.Bold))
        self.emotion_label.setStyleSheet("color: #00ff00; padding: 20px;")
        right_layout.addWidget(self.emotion_label)
        
        # Confidence display
        self.confidence_label = QLabel('Confidence: 0%')
        self.confidence_label.setFont(QFont('Arial', 16))
        self.confidence_label.setStyleSheet("color: #ffaa00; padding: 10px;")
        right_layout.addWidget(self.confidence_label)
        
        # Emotion scores
        self.scores_label = QLabel('Emotion Scores:\n' + '\n'.join([f"{e}: 0%" for e in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']]))
        self.scores_label.setFont(QFont('Courier', 10))
        self.scores_label.setStyleSheet("color: #cccccc; padding: 10px; background-color: #2a2a3e; border-radius: 5px;")
        right_layout.addWidget(self.scores_label)
        
        # Emotion history plot
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.patch.set_facecolor('#1e1e2e')
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        layout.addLayout(left_layout, 2)
        layout.addLayout(right_layout, 1)
        
        main_widget.setLayout(layout)
        
        self.detection_thread = None
        
    def start_detection(self):
        self.detection_thread = EmotionDetectionThread(self.model, self.face_cascade)
        self.detection_thread.frame_processed.connect(self.update_frame)
        self.detection_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
    def stop_detection(self):
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def update_frame(self, frame, emotion, confidence, scores):
        # Update emotion history
        self.emotion_history.append(emotion)
        
        # Convert frame for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaledToWidth(640))
        
        # Update labels
        self.emotion_label.setText(f'🎭 {emotion}')
        self.confidence_label.setText(f'Confidence: {confidence*100:.1f}%')
        
        # Update scores
        scores_text = 'Emotion Scores:\n'
        for emotion_name, score in scores.items():
            scores_text += f'{emotion_name}: {score*100:.1f}%\n'
        self.scores_label.setText(scores_text)
        
        # Update history plot
        self.update_emotion_history()
    
    def update_emotion_history(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        emotions_list = list(self.emotion_history)
        if emotions_list:
            from collections import Counter
            emotion_counts = Counter(emotions_list)
            
            ax.bar(emotion_counts.keys(), emotion_counts.values(), color='#0066cc')
            ax.set_facecolor('#1e1e2e')
            ax.tick_params(colors='white')
            ax.set_ylabel('Count', color='white')
            ax.set_title('Emotion History', color='white')
            
            # Rotate x labels
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        
        self.canvas.draw()
    
    def capture_screenshot(self):
        if self.video_label.pixmap():
            pixmap = self.video_label.pixmap()
            pixmap.save('emotion_detection_screenshot.png')
            print("Screenshot saved!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EmotionDetectionApp()
    ex.show()
    sys.exit(app.exec_())