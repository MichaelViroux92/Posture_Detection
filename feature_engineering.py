
import numpy as np

class FeatureEngineering:
    def __init__(self, keypoints):
        self.keypoints = np.array(keypoints)

    def get_distance(self, i, j): # helper method
        '''Calculate eucledian distance'''
        return np.linalg.norm(self.keypoints[i] - self.keypoints[j])

    def get_angle(self, i, j, k): #helper method
        '''Calculate angle using cosine and dotproduct'''
        #Calculate vectors from point: keypoint[j]
        vec1 = np.array(self.keypoints[i]) - np.array(self.keypoints[j])
        vec2 = np.array(self.keypoints[k]) - np.array(self.keypoints[j])
        #Calculate magnitudes of these 2 vectors
        vec1_magn = np.linalg.norm(vec1)
        vec2_magn = np.linalg.norm(vec2)
        #Calculate dot product of these 2 vectors
        dot_product = np.dot(vec1, vec2)
        #Calculate magnitude product
        magnitude = vec1_magn * vec2_magn
        #Calculate cosine
        cosine = dot_product / magnitude
        #Get angle in radians
        angle_radians = np.arccos(cosine)
        #Get angle in degrees
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    def normalize(self, center_kp_index): # helper method
        '''Normalize coordinates around center keypoint (0, 0)'''
        #Subtract center_coordinate from each keypoint.
        center_coord = self.keypoints[center_kp_index]
        self.normalized_keypoints = self.keypoints - center_coord

        return self.normalized_keypoints

    def flatten_1D(self, normalized=True, center_kp_index):
        '''Create 1D arrays as feed for ML model'''
        coords =  self.normalize(center_kp_index) if normalized=True else self.keypoints
        flattened_array = coords.flatten()
        
        return flattened_array

    def get_distance_spine(self, spine_indices):
        '''Use get_distance method to specifically calculate distances between consecutive keypoints of the spine'''
        spine_distances = [self.get_distance(spine_indices[i], spine_indices[i+1]) for i in range(len(spine_indices)-1)]

        return spine_distances

    def get_angle_spine(self, spine_indices):
        '''Use get_angle method to specifically calculate angles between consecutive keypoints of the spine'''
        spine_angles = [self.get_angle(spine_indices[i-1], spine_indices[i], spine_indices[i+1]) for i in range(1, len(spine_indices)-2)]

        return spine_angles