import os
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


class Annotater:
    def __init__(self, raw_images_folder_path, labels_folder_path):
        self.labels_folder_path = labels_folder_path
        self.raw_images_folder_path = raw_images_folder_path
        self.annotated_data = {}
        self.image_filenames = []
        self.df_annotated_images = None
        self.label_options = {
            'head': {0: 'Forward head', 1: 'Neutral head'},
            'neck': {0: 'Flexion', 1: 'Neutral', 2: 'Extension'},
            'shoulder': {0: 'Protracted', 1: 'Neutral'},
            'thoraric_spine': {0: 'Neutral', 1: 'Hyperkyfosis'},
            'lumbar_spine': {0: 'Neutral', 1: 'Hyperlordosis', 2: 'Flat back'}
        }

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

    '''
    def validate_input(self):
        input_values_head = [-1, 'Forward head', 0, 'Neutral head']
        input_values_neck = [-1, 'Flexion', 0, 'Neutral', 1, 'Extension']
        input_values_shoulder = [-1, 'Protracted', 0, 'Neutral']
        input_values_thoraric_spine = [0, 'Neutral', 1, 'Hyperkyfosis']
        input_values_lumbar_spine = [0, 'Neutral', 1, 'Hyperlordosis']
        annotation_values = {
                    'head':input_values_head,
                    'neck': input_values_neck,
                    'shoulder': input_values_shoulder,
                    'thoraric_spine': input_values_thoraric_spine,
                    'lumbar_spine': input_values_lumbar_spine,
        }
        
        
        for body_part, annotation_value in annotation_values.items():
            while True:
                user_input = int(input('Annotate '))
                if int(input()) != parameters[i]:
                    print("Wrong input, enter again")
                else:
                    continue
        '''
                    

    def validate_input(self, prompt, valid_options):
        while True:
            try:
                user_input = int(input(prompt))
                if user_input in valid_options:
                    return user_input
                else:
                    print(f"Invalid input. Choose from: {list(valid_options.keys())}")
            except ValueError:
                print("Please enter a valid integer.")


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
                    continue

                # Get annotations using validation
                annotate_head = self.validate_input("Annotate head position (0: Forward head, 1: Neutral head): ", self.label_options['head'])
                annotate_neck = self.validate_input("Annotate neck position (0: Flexion, 1: Neutral, 2: Extension): ", self.label_options['neck'])
                annotate_shoulder = self.validate_input("Annotate shoulder position (0: Protracted, 1: Neutral): ", self.label_options['shoulder'])
                annotate_thoraric_spine = self.validate_input("Annotate thoracic spine (0: Neutral, 1: Hyperkyfosis): ", self.label_options['thoraric_spine'])
                annotate_lumbar_spine = self.validate_input("Annotate lumbar spine (0: Neutral, 1: Hyperlordosis, 2: Flat back): ", self.label_options['lumbar_spine'])
                    
                self.annotated_data[image_filename] = [annotate_head, annotate_neck, annotate_shoulder, annotate_thoraric_spine, annotate_lumbar_spine]
                
                previous_status = f"Image annotated"

                    

    def save_annotated_images(self):
        columns = ['Head', 'Neck', 'Shoulder', 'Thoraric_Spine', 'Lumbar_Spine']
        self.df_annotated_images = pd.DataFrame.from_dict(self.annotated_data, orient='index', columns=columns)
        self.df_annotated_images = self.df_annotated_images.reset_index().rename(columns={'index': 'image_filename'})
        save_path = os.path.join(self.labels_folder_path, 'labels.csv')
        self.df_annotated_images.to_csv(save_path, encoding='utf-8', index=False)
        print(self.df_annotated_images)

# TODO: Input validatie (opnieuw invoeren als input != 0/1)
# TODO: Nieuwe afbeeldingen: hoe flow dan? Reset renaming?


