import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Download pose landmarker model if not exists
model_path = 'pose_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading pose landmarker model...")
    import urllib.request
    url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    urllib.request.urlretrieve(url, model_path)

# Initialize MediaPipe Pose
# Define pose connections for visualization
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right face
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm and hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm and hand
    (11, 23), (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg and foot
    (12, 24), (24, 26), (26, 28), (28, 30), (28, 32)   # Right leg and foot
]

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# Open video file
video_path = 'Virus Scan Alert.mp4'
cap = cv2.VideoCapture(video_path)
# Get video properties and swap width/height for rotation
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# Swap width and height since we'll rotate the video
width, height = height, width

# Initialize variables for FPS calculation and frame tracking
prev_time = 0
paused = False
current_frame_number = 0

# Function to draw MediaPipe pose
def draw_mediapipe_pose(frame, detection_result, width, height):
    annotated_image = frame.copy()
    if detection_result.pose_landmarks:
        for landmark in detection_result.pose_landmarks:
            # Draw landmarks with different colors based on body parts
            for idx, landmark_point in enumerate(landmark):
                x = int(landmark_point.x * width)
                y = int(landmark_point.y * height)
                
                # Color coding based on body parts
                if idx <= 10:  # Face landmarks
                    color = (255, 200, 0)  # Light blue
                elif idx <= 22:  # Upper body
                    color = (0, 255, 0)  # Green
                else:  # Lower body
                    color = (255, 0, 0)  # Red
                
                # Draw landmark point
                cv2.circle(annotated_image, (x, y), 4, color, -1)
                # Add landmark number for debugging
                cv2.putText(annotated_image, str(idx), (x+5, y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw connections between landmarks
            for connection in POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = (
                    int(landmark[start_idx].x * width),
                    int(landmark[start_idx].y * height)
                )
                end_point = (
                    int(landmark[end_idx].x * width),
                    int(landmark[end_idx].y * height)
                )
                
                cv2.line(annotated_image, start_point, end_point, (102, 204, 255), 2)
    return annotated_image

# Create output writer (uncomment to save output)
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Store the last frame for pause state
last_frame = None

while cap.isOpened():
    # Handle pause/resume with spacebar
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        paused = not paused
    elif key == ord('q'):
        break
    
    # Handle backward step
    if key == ord('b') and current_frame_number > 0:
        current_frame_number -= 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
        ret, frame = cap.read()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            last_frame = frame.copy()

    # Get and process new frame only if not paused or stepping forward
    if not paused or key == ord('n'):
        ret, frame = cap.read()
        if not ret:
            break
        # Rotate new frame 90 degrees clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        last_frame = frame.copy()
        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    else:
        # Use the last frame when paused
        frame = last_frame.copy()
    
    # Calculate FPS
    current_time = cv2.getTickCount()
    if prev_time > 0:
        fps_actual = cv2.getTickFrequency() / (current_time - prev_time)
    else:
        fps_actual = 0
    prev_time = current_time
    
    # Convert frame to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect pose
    detection_result = detector.detect(mp_image)
    
    # Draw pose landmarks
    annotated_image = draw_mediapipe_pose(rgb_frame, detection_result, width, height)
    
    # Convert back to BGR for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    # Config text properties
    text_color = (0, 255, 255) 
    fps_scale = 1.5  
    guide_scale = 2 
    thickness = 3
    
    # Add black outline for better visibility
    def put_text_with_outline(img, text, pos, scale, color, thickness):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    
    # FPS
    put_text_with_outline(annotated_image, f'FPS: {fps_actual:.1f}', (20, 50),
                         fps_scale, text_color, thickness)
    
    # Frame tools
    guide_text = 'Space: Pause/Resume  N: Next  B: Back  Q: Quit'
    text_size = cv2.getTextSize(guide_text, cv2.FONT_HERSHEY_SIMPLEX, guide_scale, thickness)[0]
    text_x = width - text_size[0] - 20  
    put_text_with_outline(annotated_image, guide_text,
                         (text_x, 50), guide_scale, text_color, thickness)
    
    # Add confidence scores if landmarks detected
    if detection_result.pose_landmarks:
        for pose_idx, landmark in enumerate(detection_result.pose_landmarks):
            if hasattr(detection_result, 'pose_world_landmarks_confidence'):
                confidence = detection_result.pose_world_landmarks_confidence[pose_idx]
                put_text_with_outline(annotated_image, f'Confidence: {confidence:.2f}',
                                    (20, 100), fps_scale, text_color, thickness)
    
    # Display result
    cv2.imshow('Pose Detection', annotated_image)
    
    # Save output (uncomment to save)
    # out.write(annotated_image)

cap.release()
# out.release()  # Uncomment if saving output
cv2.destroyAllWindows()
