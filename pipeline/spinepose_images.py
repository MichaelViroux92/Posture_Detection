import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from spinepose import SpinePoseEstimator
import json

class SpineKeypointProcessor:
    def __init__(self, images_folder_path, coord_folder_path, device='cpu'):
        self.device = device
        self.images_folder_path = images_folder_path
        self.coord_folder_path = coord_folder_path
        self.estimator = SpinePoseEstimator(device=self.device)
        self.spine_indices = [17, 18, 19, 26, 27, 28, 29, 30, 35, 36]
        self.image_outputs = []

    def process_images(self):
        self.keypoint_data = {}
        image_filenames = sorted(os.listdir(self.images_folder_path))
        for image_filename in image_filenames:
            if not image_filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            image_path = os.path.join(self.images_folder_path, image_filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Image not found at {image_path}, skipping.")
                continue

            bboxes = self.estimator.detect(image)
            if bboxes is None or bboxes.shape[0] == 0:
                print(f"No people detected in {image_filename}, skipping.")
                continue

            keypoints, scores = self.estimator.estimate(image, bboxes)

            spine_keypoints = []
            for idx in self.spine_indices:
                x_coord, y_coord = keypoints[0, idx]
                spine_keypoints.append([x_coord, y_coord])

            self.image_outputs.append((image, keypoints, scores, image_filename))
            self.keypoint_data[image_filename] = spine_keypoints

        return keypoints


    def visualize_keypoints(self, index=0, show=True, save_path=None):
        """Visualize keypoints on an image from the processed set."""
        if index >= len(self.image_outputs):
            print(f"No image at index {index}.")
            return

        image, keypoints, scores, image_filename = self.image_outputs[index]

        try:
            visualized = self.estimator.visualize(image, keypoints, scores)
        except cv2.error as e:
            print(f"Error visualizing {image_filename}: {e}")
            return

        if save_path:
            cv2.imwrite(save_path, visualized)
            print(f"Visualized output saved to {save_path}")

        if show:
            self.plot_image(visualized, title=f"Keypoints: {image_filename}")

    def plot_image(self, image_bgr, title="Spine Keypoints Detection"):
        """Convert BGR image to RGB and display using matplotlib."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(title)
        plt.show()

    def save_keypoints(self):
        json_path = os.path.join(self.coord_folder_path, 'spine_keypoints.json')
        with open(json_path, 'w') as f:
            json.dump(self.keypoint_data, f, indent=4)

        

# Optional: visualize the first image with keypoints
# processor.visualize_keypoints(index=0, save_path='output_0.jpg', show=True)


    
