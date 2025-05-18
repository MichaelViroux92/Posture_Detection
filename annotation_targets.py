import os
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


class Annotater:
    def __init__(self, raw_images_folder_path):
        self.raw_images_folder_path = raw_images_folder_path
        self.annotated_data = {}
        self.image_filenames = []
        self.df_annotated_images = None

    def rename_images(self):
        image_filenames = sorted(os.listdir(self.raw_images_folder_path))
        for count, image_filename in enumerate(image_filenames):
            extension = os.path.splitext(image_filename)[1]
            dst = f"Image_{str(count)}{extension}"
            dst_path = os.path.join(self.raw_images_folder_path, dst)
            src_path = os.path.join(self.raw_images_folder_path, image_filename)

            # Only rename if destination doesn't already exist
            if src_path != dst_path and not os.path.exists(dst_path):
                os.rename(src_path, dst_path)
            else:
                continue
        self.image_filenames = sorted(os.listdir(self.raw_images_folder_path))

    def annotate_images(self):
        self.annotated_data = {}
        previous_status = None
        image_filenames = sorted(os.listdir(self.raw_images_folder_path))
        for image_filename in image_filenames:
            if image_filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                raw_image_path = os.path.join(self.raw_images_folder_path, image_filename)
                image = mpimg.imread(raw_image_path)

                clear_output(wait=True)

                if previous_status:
                    print(previous_status)

                print(self.annotated_data)
                print(self.df_annotated_images)


                plt.imshow(image)
                plt.axis('off')
                plt.title(image_filename)
                plt.show()

                keep_image = input('Keep this image? (y/n): ')

                plt.close()

                if keep_image.lower() == 'n':
                    os.remove(raw_image_path)
                    previous_status = f"Image deleted"
                else:
                    annotate_shoulder = int(input('Annotate shoulder position (-1, 0, 1): '))
                    annotate_neck = int(input('Annotate neck position (-1, 0, 1): '))
                    annotate_thoraric_spine = int(input('Annotate thoraric spine (0, 1): '))
                    annotate_lumbar_spine = int(input('Annotate lumar spine (0, 1): '))
                    annotate_pelvis = int(input('Annotate pelvis position (0, 1): '))
                    
                    self.annotated_data[image_filename] = [annotate_shoulder, annotate_neck, annotate_thoraric_spine, annotate_lumbar_spine, annotate_pelvis]

                    previous_status = f"Image annotated"

                    

    def save_annotated_images(self):
        columns = ['Shoulder', 'Neck', 'Thoraric_Spine', 'Lumbar_Spine', 'Pelvis']
        self.df_annotated_images = pd.DataFrame.from_dict(self.annotated_data, orient='index', columns=columns)
        self.df_annotated_images = self.df_annotated_images.reset_index().rename(columns={'index': 'image_filename'})
        self.df_annotated_images.to_csv('annotated_images', encoding='utf-8', index=False)
        print(self.df_annotated_images)

# TODO: Input validatie (opnieuw invoeren als input != 0/1)
# TODO: Nieuwe afbeeldingen: hoe flow dan? Reset renaming?


