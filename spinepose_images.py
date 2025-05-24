import cv2
from spinepose import SpinePoseEstimator
import matplotlib.pyplot as plt
import os

def main():
    # Initialize estimator to run on CPU
    estimator = SpinePoseEstimator(device='cpu')

    # Adjust the image path to your actual image location
    image_path = os.path.join(os.getcwd(), 'data', 'raw_images', 'archive', 'images', 'train', 'Image_0.jpg')

    # Read image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Step 1: Detect people bounding boxes in the image
    bboxes = estimator.detect(image)

    if bboxes is None or bboxes.shape[0] == 0:
        print("No people detected in the image.")
        return

    # Step 2: Estimate spine keypoints within detected bounding boxes
    keypoints, scores = estimator.estimate(image, bboxes)

    # Step 3: Visualize keypoints on the image
    visualized = estimator.visualize(image, keypoints, scores)

    # Save the visualized output
    output_path = 'output.jpg'
    cv2.imwrite(output_path, visualized)
    print(f"Output saved to {output_path}")

    # Convert BGR (OpenCV) to RGB for matplotlib display
    visualized_rgb = cv2.cvtColor(visualized, cv2.COLOR_BGR2RGB)

    # Display the image with matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(visualized_rgb)
    plt.axis('off')
    plt.title("Spine Keypoints Detection")
    plt.show()

if __name__ == "__main__":
    main()

