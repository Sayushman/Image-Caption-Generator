# 🖼️ ImageCaption-Suggestion

A desktop GUI application that allows users to **upload and blend two images** with **gesture-controlled blending** using **MediaPipe** hand tracking.  
It then generates intelligent **image captions** using a pre-trained **transformer-based AI model**.

---

## ✨ Features

- 📸 **Upload & Blend Images** – Select two images and blend them interactively.
- 🖐️ **Hand Gesture Recognition** – Use real-time hand gestures to control the blending ratio.
- 🤖 **AI-Powered Captions** – Generate natural-language captions using a transformer model.
- 🔁 **Live Camera Feed** – View your webcam with overlayed hand landmarks.
- 🧠 **Transformer Model Integration** – Leverages a pre-trained model for intelligent caption generation.
- 🖥️ **User-Friendly GUI** – Built with Python's Tkinter for ease of use.

---

## 📷 How It Works

1. **Upload** two images using the GUI.
2. **Enable** the webcam to detect hand gestures.
3. **Adjust blending ratio** using thumbs-up or thumbs-down gestures.
4. **Click "Blend & Caption"** to generate a caption for the combined image.
5. View the **caption** and blended image directly in the app.

---

## 🛠️ Tech Stack

- Python
- Tkinter (GUI)
- OpenCV (image processing)
- MediaPipe (hand gesture detection)
- Transformers (Hugging Face - image captioning)
- PIL / NumPy

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ImageCaption-Suggestion.git
cd ImageCaption-Suggestion

2️⃣ Create a Virtual Environment (optional but recommended)

//bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3️⃣ Install Requirements

//bash
pip install -r requirements.txt

Run The Application

python app.py  # or the filename of your main script


Docker Suuport

Buils and run in Docker

docker build -t imagecaption-suggestion .
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device=/dev/video0 imagecaption-suggestion


File Structure

ImageCaption-Suggestion/
│
├── blended_images/               # Stores output images
├── transformers_objects.pkl      # Pre-trained model/tokenizer
├── app.py                        # Main application script
├── requirements.txt              # Python dependencies
└── README.md                     # Project description

🙋‍♀️ Contributions

PRs are welcome!
Enhancements like:

Adding audio narration

Improving UI aesthetics

Supporting multiple gesture styles

Feel free to fork and improve the project.
