import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame  # For playing the alarm
import threading

# Initialize pygame.mixer
pygame.mixer.init()

# Load the drowsiness model
model = load_model("model.h5")

# Load Haar Cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect drowsiness
def detect_drowsiness(frame):
    # Pre-process the image (resize, normalize, etc.)
    img = cv2.resize(frame, (80, 80))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img)
    return np.argmax(predictions)  # Return the class with the highest probability

# Function to play the alarm
def play_alarm():
    try:
        pygame.mixer.music.load('peringatan.mp3')  # Replace with your alarm file path
        pygame.mixer.music.play()
        print("Alarm playing")
    except Exception as e:
        print(f"Error playing alarm: {e}")

# Using camera
cap = cv2.VideoCapture(0)  # Replace with video file if needed
alarm_on = False  # Status of whether the alarm is sounding
blink_duration = 2  # Duration in seconds for closed eyes detection
blink_timer = 0  # Timer for counting duration of closed eyes
fps = 30  # Assumed frames per second
frames_needed = blink_duration * fps  # Calculate frames needed for 2 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Initialize status and color with default values
    status = "Tidak Diketahui"
    color = (255, 255, 255)  # White as default color for text and rectangles

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangle around face and detect drowsiness
    eyes_detected = False  # Status of eye detection
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop the face area for drowsiness detection
        face_roi = frame[y:y+h, x:x+w]

        # Detect drowsiness
        label = detect_drowsiness(face_roi)

        # Translate label into text
        if label == 0:
            status = "Tidak Bermain HP"
            if pygame.mixer.music.get_busy():  # If alarm is sounding, stop it
                pygame.mixer.music.stop()
            alarm_on = False
            blink_timer = 0  # Reset timer when not drowsy
            color = (0, 255, 0)  # Green for Not Drowsy
        else:
            status = " Bermain HP"
            color = (0, 0, 255)  # Red for Drowsy

            # Detect eyes in the face area
            eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
            if len(eyes) == 0:  # If no eyes are detected
                eyes_detected = False
                blink_timer += 1  # Increment timer each frame
                if blink_timer >= frames_needed:  # Check if the timer has reached the threshold
                    if not pygame.mixer.music.get_busy() and not alarm_on:
                        alarm_on = True
                        threading.Thread(target=play_alarm).start()  # Play alarm asynchronously
            else:
                eyes_detected = True
                blink_timer = 0  # Reset timer if eyes are detected

            # Draw rectangle around detected eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), color, 2)

    # Display status on frame
    cv2.putText(frame, f'Status: {status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show frame with status
    cv2.imshow('Drowsiness Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()