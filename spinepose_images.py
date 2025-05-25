import cv2
from spinepose import SpinePoseEstimator
import matplotlib.pyplot as plt
import os

def main():
    estimator = SpinePoseEstimator(device='cpu') # change to gpu later
    image_path = os.path.join(os.getcwd(), 'data', 'raw_images', 'archive', 'images', 'train', 'Image_0.jpg')

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    bboxes = estimator.detect(image)

    if bboxes is None or bboxes.shape[0] == 0:
        print("No people detected in the image.")
        return

    keypoints, scores = estimator.estimate(image, bboxes)
    print(keypoints)
    
    ##
    # Get only keypoints of interest(spine keypoints)
    ##

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

