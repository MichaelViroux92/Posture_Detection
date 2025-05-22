import mediapipe as mp

mp_pose = mp.solutions.pose #accesses pose solution inside Mediapipe
pose = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.5) #create instance of pose class
results = pose.process(image)