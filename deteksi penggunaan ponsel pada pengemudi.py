import cv2
import mediapipe as mp
import numpy as np
import pygame  # For playing the alarm
import threading

# Initialize pygame.mixer
pygame.mixer.init()

# Function to play the alarm
def play_alarm():
    try:
        pygame.mixer.music.load('peringatan.mp3')  # Replace with your alarm file path
        pygame.mixer.music.play()
        print("Alarm playing")
    except Exception as e:
        print(f"Error playing alarm: {e}")

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Using camera
cap = cv2.VideoCapture(0)  # Replace with video file if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of specific landmarks (e.g., wrist)
            wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
            wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)

            # Example: Display wrist coordinates
            cv2.putText(frame, f'Wrist: ({wrist_x}, {wrist_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show frame with landmarks
    cv2.imshow('Hand Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
