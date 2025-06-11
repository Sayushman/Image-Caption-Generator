# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory in container
WORKDIR /app



# Install system dependencies required for OpenCV, Tkinter, and MediaPipe
RUN apt-get update && apt-get install -y \
    python3-tk \
    bcc==0.29.1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Expose port (optional for GUI apps)
EXPOSE 8501

# Default command to run the GUI app
CMD ["python", "your_script.py"]

