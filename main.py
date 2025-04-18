import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os

class HandGestureFilterApp:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize Tkinter window
        self.root = tk.Tk()
        self.root.title("Hand Gesture Filter App")
        
        # Create video display label
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(padx=10, pady=10)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Load filter models (placeholder for actual model loading)
        self.filters = {
            2: self.vector_art_filter
        }
        
        # Start video loop
        self.update_frame()
        
    def count_fingers(self, hand_landmarks):
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tip landmarks
        thumb_tip = 4
        fingers_up = 0
        
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            fingers_up += 1
            
        for tip in finger_tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers_up += 1
                
        return fingers_up

    def ghibli_filter(self, image):
        # Cartoon effect
        # Reduce the color palette
        num_colors = 8
        div = 256 // num_colors
        image = image // div * div + div // 2
        
        # Apply bilateral filter to smooth the image while preserving edges
        smooth = cv2.bilateralFilter(image, d=9, sigmaColor=300, sigmaSpace=300)
        
        # Convert to grayscale and apply adaptive thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 9, 2
        )
        
        # Convert edges back to BGR
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine smoothed image with edges
        cartoon = cv2.bitwise_and(smooth, edges)
        
        # Enhance colors
        cartoon = cv2.convertScaleAbs(cartoon, alpha=1.2, beta=10)
        
        return cartoon

    def watercolor_anime_filter(self, image):
        # Simplified watercolor effect
        image = cv2.stylization(image, sigma_s=60, sigma_r=0.6)
        return image

    def vector_art_filter(self, image):
        # Posterization effect
        image = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
        edges = cv2.Canny(image, 100, 200)
        return cv2.bitwise_and(image, image, mask=edges)

    def korean_filter(self, image):
        # Brightening and smoothing
        image = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
        image = cv2.bilateralFilter(image, 9, 75, 75)
        return image

    def cel_shaded_filter(self, image):
        # Glitch effect
        height, width = image.shape[:2]
        
        # Create glitch effects
        num_glitches = np.random.randint(10, 20)
        
        # Make a copy of the image
        glitched = image.copy()
        
        for _ in range(num_glitches):
            # Random position and size for glitch
            x = np.random.randint(0, width - 50)
            y = np.random.randint(0, height - 15)
            w = np.random.randint(50, 100)
            h = np.random.randint(5, 15)
            
            # Random RGB channel shift
            if x + w < width and y + h < height:
                # Get the region of interest
                roi = glitched[y:y+h, x:x+w]
                
                # Randomly shift color channels
                if np.random.random() > 0.5:
                    # Shift red channel
                    roi[:, :, 2] = np.roll(roi[:, :, 2], np.random.randint(-20, 20))
                if np.random.random() > 0.5:
                    # Shift blue channel
                    roi[:, :, 0] = np.roll(roi[:, :, 0], np.random.randint(-20, 20))
        
        # Add color distortion
        glitched = cv2.add(glitched, np.array([np.random.randint(-50, 50)]))
        
        # Add scan lines
        scan_lines = np.zeros_like(image)
        for i in range(0, height, 3):
            scan_lines[i:i+1, :] = [255, 255, 255]
        glitched = cv2.addWeighted(glitched, 0.9, scan_lines, 0.1, 0)
        
        # Add noise
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        glitched = cv2.add(glitched, noise)
        
        return glitched

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                             self.mp_hands.HAND_CONNECTIONS)
                    
                    # Count fingers and apply corresponding filter
                    fingers = self.count_fingers(hand_landmarks)
                    if fingers in self.filters:
                        frame = self.filters[fingers](frame)
            
            # Convert frame for Tkinter display
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_frame)

    def run(self):
        self.root.mainloop()

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    app = HandGestureFilterApp()
    app.run()