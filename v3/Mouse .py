#!/usr/bin/env python3
"""
hand_mouse.py
==============

Move the mouse cursor with your bare hand using a webcam.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# Grab the current screen size – PyAutoGUI returns (width, height)
screen_width, screen_height = pyautogui.size()

# Smoothing settings – keep last N positions and average them
SMOOTHING = 5                # larger = smoother, but slower to respond
history = deque(maxlen=SMOOTHING)

# Optional: map only a specific ROI in the camera frame
# For example, set roi = (x, y, w, h) to map a sub‑rectangle to the whole screen.
# Set roi = None to use the whole frame.
roi = None  # (200, 200, 400, 400)  # Example values

# ------------------------------------------------------------------
# MediaPipe hand detector
# ------------------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def screen_coords(norm_x, norm_y, frame_w, frame_h):
    """
    Convert MediaPipe normalized (0–1) coords to screen pixels.
    """
    if roi:
        # If ROI is defined, map it to the full screen
        rx, ry, rw, rh = roi
        norm_x = (norm_x - rx / frame_w) / (rw / frame_w)
        norm_y = (norm_y - ry / frame_h) / (rh / frame_h)

    # Clamp between 0 and 1
    norm_x = np.clip(norm_x, 0.0, 1.0)
    norm_y = np.clip(norm_y, 0.0, 1.0)

    # Convert to pixel coords on screen
    return int(norm_x * screen_width), int(norm_y * screen_height)

def average_history():
    """Return the average of the recent positions."""
    if not history:
        return None
    xs, ys = zip(*history)
    return int(np.mean(xs)), int(np.mean(ys))

def is_pinching(landmarks, frame_w, frame_h, threshold=0.02):
    """
    Simple pinch detection:
    distance between thumb tip (4) and index finger tip (8) < threshold
    """
    thumb = landmarks[4]
    index = landmarks[8]
    dist = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
    return dist < threshold

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)          # 0 = default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pinching = False          # True while the hand is pinching
last_press = 0
CLICK_COOLDOWN = 0.3  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Mirror so that movement feels natural
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # --------------------------------------------------------------
        # Draw hand annotations (optional, but handy for debugging)
        # --------------------------------------------------------------
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            # Use the *first* hand detected (since max_num_hands=1)
            lm = results.multi_hand_landmarks[0].landmark

            # Grab the tip of the index finger (landmark 8)
            index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_screen, y_screen = screen_coords(index_tip.x, index_tip.y, w, h)

            # Add to history for smoothing
            history.append((x_screen, y_screen))

            # Smooth by averaging
            avg_pos = average_history()
            if avg_pos:
                pyautogui.moveTo(*avg_pos, duration=0.001)  # tiny duration for responsive move

            # Optional: click on pinch
            if pinching and is_pinching(lm,w,h):
                # If we just started pinching, press the button
                if not pyautogui.mouseDown():  # pyautogui.mouseDown() returns False if already down
                    pyautogui.mouseDown()  # press left button
                    last_press = time.time()
            elif not pinching:
                # If we were pinching before but now stopped, release
                if pyautogui.mouseDown():
                    pyautogui.mouseUp()
        # --------------------------------------------------------------
        # Show the webcam feed (optional)
        # --------------------------------------------------------------
        cv2.imshow("Hand Cursor", frame)

        # Press 'q' or ESC to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
