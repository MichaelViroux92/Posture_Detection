import json
import numpy as np
import pandas as pd
from pipeline.feature_engineering import FeatureEngineering

class SpineFeatureBuilder:
    def __init__(self, json_path, spine_indices_order):
        """
        json_path: path to your spine keypoints JSON file
        spine_indices_order: list of original spine indices corresponding to your keypoints order in JSON
        """
        self.json_path = json_path
        self.spine_indices_order = spine_indices_order
        self.keypoints_data = None
        self.df_features = None

    def load_keypoints(self):
        with open(self.json_path, 'r') as f:
            self.keypoints_data = json.load(f)

    def build_features(self):
        features_all = []

        # zero-based indices because keypoints_list is already ordered as spine_indices_order
        spine_indices = list(range(len(self.spine_indices_order)))

        for image_name, keypoints_list in self.keypoints_data.items():
            fe = FeatureEngineering(keypoints_list)
            fe.normalize(center_kp_index=0)

            distances = fe.get_distance_spine(spine_indices)
            angles = fe.get_angle_spine(spine_indices)
            flattened = fe.flatten_1D()

            combined_features = np.concatenate([flattened, distances, angles])

            # Create column names
            feature_names = []
            for idx in self.spine_indices_order:
                feature_names.extend([f'X_{idx}', f'Y_{idx}'])
            for i in range(len(self.spine_indices_order) - 1):
                feature_names.append(f'dist_{self.spine_indices_order[i]}_{self.spine_indices_order[i+1]}')
            for i in range(1, len(self.spine_indices_order) - 1):
                feature_names.append(f'angle_{self.spine_indices_order[i-1]}_{self.spine_indices_order[i]}_{self.spine_indices_order[i+1]}')

            features_all.append([image_name] + combined_features.tolist())

        columns = ['image_filename'] + feature_names
        self.df_features = pd.DataFrame(features_all, columns=columns)

    def get_features_dataframe(self):
        if self.df_features is None:
            raise ValueError("Features not built yet. Run build_features() first.")
        return self.df_features
