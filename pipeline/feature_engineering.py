
import numpy as np


class FeatureEngineering:
    def __init__(self, keypoints):
        self.keypoints = np.array(keypoints)

    def get_distance(self, i, j):
        '''Calculate Euclidean distance between two keypoints'''
        return np.linalg.norm(self.keypoints[i] - self.keypoints[j])

    def get_angle(self, i, j, k):
        '''Calculate angle (in degrees) formed at keypoint j using i-j-k'''
        vec1 = self.keypoints[i] - self.keypoints[j]
        vec2 = self.keypoints[k] - self.keypoints[j]

        vec1_magn = np.linalg.norm(vec1)
        vec2_magn = np.linalg.norm(vec2)

        if vec1_magn == 0 or vec2_magn == 0:
            return 0.0  # Avoid division by zero

        cosine = np.dot(vec1, vec2) / (vec1_magn * vec2_magn)
        cosine = np.clip(cosine, -1.0, 1.0)  # Handle rounding errors

        angle_radians = np.arccos(cosine)
        return np.degrees(angle_radians)

    def normalize(self, center_kp_index):
        '''Return keypoints normalized around a center keypoint'''
        center_coord = self.keypoints[center_kp_index]
        self.keypoints = self.keypoints - center_coord
        return self.keypoints

    def flatten_1D(self, center_kp_index, normalized=True):
        '''Return keypoints as flattened 1D array'''
        return self.keypoints.flatten()

    def get_distance_spine(self, spine_indices):
        '''Distances between consecutive spine keypoints'''
        return [self.get_distance(spine_indices[i], spine_indices[i+1]) 
                for i in range(len(spine_indices) - 1)]

    def get_angle_spine(self, spine_indices):
        '''Angles formed between every triplet of consecutive spine keypoints'''
        return [self.get_angle(spine_indices[i - 1], spine_indices[i], spine_indices[i + 1]) 
                for i in range(1, len(spine_indices) - 1)]
