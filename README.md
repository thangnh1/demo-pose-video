# Demo Pose Video

A Python application that demonstrates real-time pose detection using MediaPipe. The application processes video files and displays pose landmarks with color-coded body parts.

## Features

- Real-time pose detection and visualization
- Color-coded body parts (face, upper body, lower body)
- Frame-by-frame navigation controls:
  - Space: Pause/Resume
  - N: Step forward one frame
  - B: Step backward one frame
  - Q: Quit
- FPS display and confidence score
- Automatic model download

## Requirements

- Python 3.12+
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/thangnh1/demo-pose-video.git
cd demo-pose-video
```

2. Create and activate a virtual environment:

```bash
python -m venv .
source bin/activate  # On Windows: .\Scripts\activate
```

3. Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

## Usage

1. Place your video file in the project directory
2. Update the video path in `test.py` if needed
3. Run the application:

```bash
python test.py
```

The pose landmarker model will be automatically downloaded on first run.
