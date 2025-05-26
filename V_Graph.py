# Object Graph Feature Extraction for Rotated Bounding Boxes

import json
import numpy as np
import os


# Function to load JSON data
def load_json(filepath):
    with open (filepath, 'r') as f:
        return json.load (f)


# Function to calculate area of rotated bounding box
def calculate_bbox_area(bbox):
    _, _, width, height, _, _ = bbox
    return width * height


# Function to build adjacency and degree matrices
def build_matrices(json_file_path, num_classes=20):
    data = load_json (json_file_path)
    num_objects = len (data)

    # Initialize adjacency and degree matrices
    adjacency_matrix = np.zeros ((num_classes, num_classes), dtype=float)
    degree_matrix = np.zeros ((num_classes, num_classes), dtype=float)

    # Class index mapping (example)
    label_mapping = {"plane": 0, "car": 1, "building": 2}  # Extend as needed

    # Iterate through all objects and populate the matrices
    for i in range (num_objects):
        obj1 = data[i]
        if obj1['label'] not in label_mapping:
            continue
        class_idx_1 = label_mapping[obj1['label']]

        # Calculate degree contribution (number of objects * area * probability)
        bbox_area = calculate_bbox_area (obj1['bbox'])
        probability = abs (obj1['bbox'][-1])  # Example to get probability
        degree_matrix[class_idx_1, class_idx_1] += (bbox_area * probability)

        # Calculate adjacency relationships
        for j in range (i + 1, num_objects):
            obj2 = data[j]
            if obj2['label'] not in label_mapping:
                continue
            class_idx_2 = label_mapping[obj2['label']]

            # Relative distance can be calculated here between object pairs
            x1, y1, _, _, _, _ = obj1['bbox']
            x2, y2, _, _, _, _ = obj2['bbox']
            distance = np.sqrt ((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Arbitrary calculation for adjacency weight based on distance
            if distance > 0:
                weight = 1.0 / distance
            else:
                weight = 0  # Distance should not be zero unless objects overlap perfectly

            adjacency_matrix[class_idx_1, class_idx_2] += weight
            adjacency_matrix[class_idx_2, class_idx_1] += weight

    # Set diagonal of adjacency matrix to zero
    np.fill_diagonal (adjacency_matrix, 0)
    return adjacency_matrix, degree_matrix

if __name__ == "__main__":
    # Example usage
    json_path = "./det_json/airport_2.json"
    adjacency_matrix, degree_matrix = build_matrices (json_path)

    print ("Adjacency Matrix:")
    print (adjacency_matrix)

    print ("\nDegree Matrix:")
    print (degree_matrix)
