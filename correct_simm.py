import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, GenerationConfig
import torch
import pickle
import os
from datetime import datetime
import mediapipe as mp
import threading
import traceback

# Load the objects from the pickle file
try:
    with open('transformers_objects.pkl', 'rb') as f:
        objects_dict = pickle.load(f)
except Exception as e:
    print(f"Error loading transformers objects: {e}")
    traceback.print_exc()

model = objects_dict.get('model')
feature_extractor = objects_dict.get('feature_extractor')
tokenizer = objects_dict.get('tokenizer')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4

# Function to predict captions for images
def predict_step(image_path):
    try:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        encoding = feature_extractor(images=[i_image], return_tensors='pt')
        pixel_values = encoding.pixel_values.to(device)

        # Define a new generation configuration
        generation_config = GenerationConfig(max_length=max_length, num_beams=num_beams)

        # Generate output ids
        output_ids = model.generate(pixel_values, generation_config=generation_config)

        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return caption
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        return "Error in generating caption"

# Function to load an image
def load_image(label):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        label.file_path = file_path
        label.config(text="Image uploaded")

# Function to blend images and generate caption
def blend_images(blend_ratio1=0.7, blend_ratio2=0.3):
    if not (hasattr(image_label1, 'file_path') and hasattr(image_label2, 'file_path')):
        messagebox.showerror("Error", "Please upload both images")
        return

    # Show progress bar
    progress_bar.start()

    def process_blending():
        try:
            image1 = Image.open(image_label1.file_path)
            image2 = Image.open(image_label2.file_path)

            img1 = np.array(image1)
            img2 = np.array(image2)

            # Resize images to the same size
            img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # Ensure both images have the same number of channels
            if len(img1.shape) == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if len(img2_resized.shape) == 2:
                img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2BGR)

            # Check for number of channels again and convert if necessary
            if img1.shape[2] != img2_resized.shape[2]:
                if img1.shape[2] > img2_resized.shape[2]:
                    img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
                    img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2BGR)
                else:
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

            blended_img = cv2.addWeighted(img1, blend_ratio1, img2_resized, blend_ratio2, 0)
            blended_image = Image.fromarray(blended_img)
            blended_image.thumbnail((250, 250))
            blended_img_tk = ImageTk.PhotoImage(blended_image)

            blended_image_label.configure(image=blended_img_tk)
            blended_image_label.image = blended_img_tk

            # Create the directory to save blended images if it doesn't exist
            blended_images_dir = 'blended_images'
            if not os.path.exists(blended_images_dir):
                os.makedirs(blended_images_dir)

            # Generate a unique filename using the current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blended_image_path = os.path.join(blended_images_dir, f"blended_image_{timestamp}.jpg")

            # Convert to RGB mode before saving if the image is in RGBA mode
            if blended_image.mode == 'RGBA':
                blended_image = blended_image.convert('RGB')

            # Save blended image to the specified path
            blended_image.save(blended_image_path)

            # Generate caption for the blended image
            caption = predict_step(blended_image_path)
            caption_label.config(text=f"Generated Caption: {caption}")

        except Exception as e:
            print(f"An error occurred during blending: {e}")
            traceback.print_exc()
        finally:
            # Stop progress bar
            progress_bar.stop()

    threading.Thread(target=process_blending).start()

# Tkinter GUI setup
root = tk.Tk()
root.title("Image Blending and Captioning")
root.configure(bg='#1e1e1e')

# Define styles
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), padding=10)
style.configure('TLabel', font=('Helvetica', 12), background='#1e1e1e', foreground='white')
style.configure('TFrame', background='#1e1e1e')
style.configure('TLabelFrame', font=('Helvetica', 12), background='#1e1e1e', foreground='white')

# Main frame
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# Split the main frame into left and right frames
left_frame = ttk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

right_frame = ttk.Frame(main_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

# Add a border line between left and right frames
border_line = ttk.Separator(main_frame, orient=tk.VERTICAL)
border_line.pack(side=tk.LEFT, fill=tk.Y, padx=5)

# Title
title_label = ttk.Label(left_frame, text="Image Blending and Captioning", font=("Helvetica", 18, 'bold'))
title_label.pack(pady=20)

# Upload image frame 1
upload_frame1 = ttk.LabelFrame(left_frame, text="Upload Image 1", padding="10")
upload_frame1.pack(pady=10, fill="x")

upload_btn1 = ttk.Button(upload_frame1, text="Browse files", command=lambda: load_image(image_label1))
upload_btn1.pack(side=tk.RIGHT, padx=10)
image_label1 = ttk.Label(upload_frame1, text="Drag and drop file here\nLimit 200MB per file • JPG, JPEG, PNG")
image_label1.pack(side=tk.LEFT, padx=10)

# Upload image frame 2
upload_frame2 = ttk.LabelFrame(left_frame, text="Upload Image 2", padding="10")
upload_frame2.pack(pady=10, fill="x")

upload_btn2 = ttk.Button(upload_frame2, text="Browse files", command=lambda: load_image(image_label2))
upload_btn2.pack(side=tk.RIGHT, padx=10)
image_label2 = ttk.Label(upload_frame2, text="Drag and drop file here\nLimit 200MB per file • JPG, JPEG, PNG")
image_label2.pack(side=tk.LEFT, padx=10)

# Progress bar
progress_bar = ttk.Progressbar(left_frame, mode='indeterminate')
progress_bar.pack(pady=10, fill=tk.X)

# Blend and caption button
blend_btn = ttk.Button(left_frame, text="Blend Images and Generate Caption", command=lambda: blend_images(blend_ratio1, blend_ratio2))
blend_btn.pack(pady=20)

# Label to display blended image and caption
blended_image_label = ttk.Label(left_frame)
blended_image_label.pack(pady=10)
caption_label = ttk.Label(left_frame, text="Generated Caption: ")
caption_label.pack(pady=10)

# Label to display the camera feed
camera_feed_label = ttk.Label(right_frame)
camera_feed_label.pack(pady=10, fill=tk.BOTH, expand=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

blend_ratio1 = 0.7
blend_ratio2 = 0.3

def recognize_gesture(frame):
    global blend_ratio1, blend_ratio2
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate vertical distance between thumb tip and index finger tip
                vertical_distance = thumb_tip.y - index_tip.y

                # Adjust blend ratios based on vertical thumb movement
                if vertical_distance > 0.1:  # Thumbs up
                    blend_ratio1 = min(blend_ratio1 + 0.05, 1.0)
                    blend_ratio2 = max(1.0 - blend_ratio1, 0.0)
                    print(f"Blend Ratios - Image 1: {blend_ratio1}, Image 2: {blend_ratio2}")
                elif vertical_distance < -0.1:  # Thumbs down
                    blend_ratio1 = max(blend_ratio1 - 0.05, 0.0)
                    blend_ratio2 = min(1.0 - blend_ratio1, 1.0)
                    print(f"Blend Ratios - Image 1: {blend_ratio1}, Image 2: {blend_ratio2}")

        return "no_gesture"
    except Exception as e:
        print(f"An error occurred during gesture recognition: {e}")
        traceback.print_exc()

def update_camera_feed():
    try:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            gesture = recognize_gesture(frame)
            if gesture == "thumbs_up" or gesture == "thumbs_down":
                blend_images(blend_ratio1, blend_ratio2)

            # Convert frame to ImageTk format and update the label
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_feed_label.imgtk = imgtk
            camera_feed_label.configure(image=imgtk)

            # Break the loop on 'Esc' key press
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
    except Exception as e:
        print(f"An error occurred during camera feed update: {e}")
        traceback.print_exc()

# Run camera feed update in a separate thread to keep the GUI responsive
camera_thread = threading.Thread(target=update_camera_feed, daemon=True)
camera_thread.start()

try:
    root.mainloop()
except Exception as e:
    print(f"An error occurred in the main loop: {e}")
    traceback.print_exc()
