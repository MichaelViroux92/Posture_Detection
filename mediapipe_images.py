import mediapipe as mp
import cv2
import os

mp_pose = mp.solutions.pose #accesses pose solution inside Mediapipe
pose = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.5) #create instance of pose class

mp_drawing = mp.solutions.drawing_utils

image_path = os.path.join(os.getcwd(), 'data', 'raw_images', 'archive', 'images', 'train', 'Image_0.jpg')
image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = pose.process(image_rgb)

if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    print("Pose landmarks detected.")
else:
    print("No pose landmarks detected.")

# Display the image
cv2.imshow('Pose Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()