## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    # Define two vectors
    vector1 = np.array([1, 0, 0])
    vector2 = np.array([0, 1, 0])
    
    # Calculate cosine similarity
    result = cosine_similarity(vector1, vector2)
    
    # The expected cosine similarity for orthogonal vectors is 0
    expected_result = 0.0
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_nearest_neighbor():
    # Define a target vector and a matrix of vectors
    target_vector = np.array([1, 1, 1])
    vectors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    
    # Calculate the nearest neighbor
    result = nearest_neighbor(target_vector, vectors)
    
    # The nearest neighbor should be the last vector, which is at index 3
    expected_index = 3
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
