import numpy as np

def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return None  # Return None explicitly if less than two points

    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)

    # Interpolation from the distance 'L' in the range [0, 1] to the range [0, 1000]
    distance = np.interp(L, [0, 1], [0, 1000])
    return distance
