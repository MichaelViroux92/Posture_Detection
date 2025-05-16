import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg' if you use Qt


script_dir = os.path.dirname(os.path.abspath(__file__))
raw_images_folder_path = os.path.join(script_dir, 'data', 'raw_images', 'archive', 'images', 'train')

for image_file_name in os.listdir(raw_images_folder_path):
    if image_file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        raw_image_path = os.path.join(raw_images_folder_path, image_file_name)
        image = mpimg.imread(raw_image_path)

        # Show the image and wait for input BEFORE closing
        plt.imshow(image)
        plt.axis('off')  # Optional: hides axis ticks
        plt.title(image_file_name)  # Optional: show filename
        plt.show(block=False)

        user_input = input('Keep this image? (y/n): ')

        # Close after input
        plt.close()

        if user_input.lower() == 'n':
            print("Image deleted")
            # os.remove(raw_image_path)
        else:
            print("Image kept")

     
