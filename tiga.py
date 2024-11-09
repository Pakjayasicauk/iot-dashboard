import cv2
import mediapipe as mp
import numpy as np
import pygame  # For playing the alarm
import threading

# Initialize pygame.mixer for the alarm
pygame.mixer.init()

# Function to play the alarm
def play_alarm():
    try:
        pygame.mixer.music.load('peringatan.mp3')  # Replace with your alarm file path
        pygame.mixer.music.play()
        print("Alarm playing")
    except Exception as e:
        print(f"Error playing alarm: {e}")

# Setup MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Setup OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Using camera for detection
cap = cv2.VideoCapture(0)  # You can replace with a video file if necessary
alarm_on = False  # To track alarm status

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert the BGR image to RGB before processing with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Detect faces using Haar Cascade
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    hand_detected_near_face = False

    # Process hand landmarks if available
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks of specific points on the hand (e.g., index finger tip)
            index_finger_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
            index_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)

            # Check proximity of the hand to the face
            for (x, y, w, h) in faces:
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Calculate distance between face center and index finger tip
                distance = np.sqrt((index_finger_tip_x - face_center_x) ** 2 + (index_finger_tip_y - face_center_y) ** 2)

                if distance < 100:  # If hand is within 100 pixels of face, assume phone usage
                    hand_detected_near_face = True
                    cv2.putText(frame, "Ponsel Terdeteksi!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Trigger alarm if not already playing
                    if not alarm_on:
                        alarm_on = True
                        threading.Thread(target=play_alarm).start()

            # Draw rectangle around the detected hand
            cv2.rectangle(frame, (index_finger_tip_x - 10, index_finger_tip_y - 10),
                          (index_finger_tip_x + 10, index_finger_tip_y + 10), (0, 255, 0), 2)

    # Reset alarm if no hand detected near face
    if not hand_detected_near_face:
        alarm_on = False
        pygame.mixer.music.stop()

    # Display the resulting frame
    cv2.imshow('Deteksi Penggunaan Ponsel', frame)

    # Exit on 'q' keyimport cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame  # For playing the alarm
import threading

# Initialize pygame.mixer
pygame.mixer.init()

# Load the CNN model for detecting phone usage
model = load_model("model.h5")  # Replace with your trained CNN model

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect phone usage using CNN model
def detect_phone_usage(frame):
    # Pre-process the image (resize, normalize, etc.)
    img = cv2.resize(frame, (80, 80))  # Resize to match input size of CNN
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(img)
    return np.argmax(predictions)  # Return the class with the highest probability

# Function to play the alarm when phone usage is detected
def play_alarm():
    try:
        pygame.mixer.music.load('peringatan.mp3')  # Replace with your alarm file path
        pygame.mixer.music.play()
        print("Alarm playing")
    except Exception as e:
        print(f"Error playing alarm: {e}")

# Using the camera for realtime detection
cap = cv2.VideoCapture(0)  # Replace with video file if needed
alarm_on = False  # Status of whether the alarm is sounding
fps = 30  # Assumed frames per second

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Initialize status and color with default values
    status = "Tidak Diketahui"
    color = (255, 255, 255)  # White as default color for text and rectangles

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    phone_detected = False  # Status of phone usage detection

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop the face area for phone usage detection
        face_roi = frame[y:y+h, x:x+w]

        # Detect if the phone is being used
        label = detect_phone_usage(face_roi)

        # Translate label into text
        if label == 0:  # 0 means no phone usage
            status = "Tidak Bermain HP"
            if pygame.mixer.music.get_busy():  # If alarm is sounding, stop it
                pygame.mixer.music.stop()
            alarm_on = False
            color = (0, 255, 0)  # Green for no phone usage
        else:  # 1 means phone is being used
            status = "Bermain HP"
            color = (0, 0, 255)  # Red for phone usage
            phone_detected = True

            # Play alarm if phone is detected and alarm is not yet on
            if not pygame.mixer.music.get_busy() and not alarm_on:
                alarm_on = True
                threading.Thread(target=play_alarm).start()  # Play alarm asynchronously

    # Display status on frame
    cv2.putText(frame, f'Status: {status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show frame with status
    cv2.imshow('Deteksi Penggunaan Ponsel', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()