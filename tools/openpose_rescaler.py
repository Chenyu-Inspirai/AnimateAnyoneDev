import cv2
import numpy as np
import sys
sys.path.append('/cephfs/SZ-AI/usr/liuchenyu/HaiLook/Moore-AnimateAnyone')
sys.path.append('/cephfs/SZ-AI/usr/liuchenyu/HaiLook/Moore-AnimateAnyone/tools')
from src.utils.util import get_fps, read_frames, save_videos_from_pil
from src.dwpose import DWposeDetector, draw_pose
from controlnet_aux.util import HWC3, resize_image
from PIL import Image
from copy import copy
import matplotlib.pyplot as plt


pairs={}
pairs["bodies"] = [
    (1,0),
    (1,2),
    (2,3),
    (3,4),
    (1,5),
    (5,6),
    (6,7),
    (1,8),
    (8,9),
    (9,10),
    (1,11),
    (11,12),
    (12,13),
    (0,14),
    (14,16),
    (0,15),
    (15,17)
    ]

pairs["hands"] = [
    (0,1),
    (1,2),
    (2,3),
    (3,4),
    (0,5),
    (5,6),
    (6,7),
    (7,8),
    (0,9),
    (9,10),
    (10,11),
    (11,12),
    (0,13),
    (13,14),
    (14,15),
    (15,16),
    (0,17),
    (17,18),
    (18,19),
    (19,20)
]

pairs["face ref indices"] = [
    (0,16),
    (2,14),
    (27,8)
]

pairs["upper face"] = [0, 1] + list(np.arange(15, 31)) + list(np.arange(36, 48))

pairs["lower face"] = list(set((np.arange(0,68))) - set(pairs["upper face"]))


def crop_and_resize(img, new_size):
    # Open an image file
    # Calculate the center point
    center = (img.width // 2, img.height // 2)
    half_width, half_height = new_size[0] // 2, new_size[1] // 2

    # Define the cropping box
    box = (
        center[0] - half_width, 
        center[1] - half_height, 
        center[0] + half_width, 
        center[1] + half_height
    )

    # Crop the image
    cropped_img = img.crop(box)

    # Resize the image
    resized_img = cropped_img.resize(new_size)

    return resized_img


def calculate_distance(keypoint1, keypoint2):
    return np.linalg.norm(keypoint1 - keypoint2)

# Function to rescale the body keypoints
def rescale_body_keypoints(pose_keypoints, reference_keypoints, body_limb_indices, fixed_index):
    scaled_keypoints = np.copy(pose_keypoints)
    prev_scale_factor = 1.0
    for limb_indices in body_limb_indices:
        (start_index, end_index) = limb_indices
        
        if pose_keypoints[start_index].min() >=0 and pose_keypoints[end_index].min() >= 0:
            # Calculate the target limb length
            target_limb_length = calculate_distance(
                pose_keypoints[start_index], pose_keypoints[end_index]
            )
            # If the keypoints is not detected, we skip this pair and use the last recorded scale
            if reference_keypoints[start_index].min() >=0 and reference_keypoints[end_index].min()>=0:
                # Calculate the reference limb length
                reference_limb_length = calculate_distance(
                    reference_keypoints[start_index], reference_keypoints[end_index]
                )
                # Calculate scale factor
                scale_factor = reference_limb_length / target_limb_length if target_limb_length != 0 else 0
            else:
                scale_factor = prev_scale_factor
            # Scale the limb in the target keypoints
            direction_vector = pose_keypoints[end_index] - pose_keypoints[start_index]
            scaled_vector = direction_vector * scale_factor
            scaled_keypoints[end_index] = scaled_keypoints[start_index] + scaled_vector
            prev_scale_factor = scale_factor
    
    # Translate the entire pose so that the fixed point matches
    translation_vector = reference_keypoints[fixed_index] - scaled_keypoints[fixed_index]
    for i in range(len(scaled_keypoints)):
        if scaled_keypoints[i].min() != -1:
            scaled_keypoints[i] = scaled_keypoints[i] + translation_vector
    
    # we still want to keep the position of target in the driving pose
    delta_x_y = scaled_keypoints[1] - pose_keypoints[1]
    for i in range(len(scaled_keypoints)):
        if scaled_keypoints[i].min() != -1:
            scaled_keypoints[i] = scaled_keypoints[i] - delta_x_y
    # scaled_keypoints = scaled_keypoints - delta_x_y

    return scaled_keypoints

def calculate_outline_scale_factors(source_keypoints, target_keypoints, outline_indices):
    # Calculate the center of the face outline for source and target
    source_center = np.mean(source_keypoints[outline_indices], axis=0)
    target_center = np.mean(target_keypoints[outline_indices], axis=0)

    # Calculate scale factors for each point in the outline
    scale_factors = np.linalg.norm(target_keypoints[outline_indices] - target_center, axis=1) / \
                    np.linalg.norm(source_keypoints[outline_indices] - source_center, axis=1)
    
    # Avoid division by zero in case of overlapping points
    scale_factors = np.where(scale_factors == np.inf, 1, scale_factors)
    
    return scale_factors, source_center, target_center

def apply_face_scaling(source_keypoints, target_keypoints, outline_indices, rest_indices, width_indices, height_indices):
    outline_scale_factors, source_center, target_center = calculate_outline_scale_factors(
        source_keypoints, target_keypoints, outline_indices
    )

    # Apply scaling to outline keypoints
    scaled_keypoints = np.copy(source_keypoints)
    for i, scale_factor in zip(outline_indices, outline_scale_factors):
        scaled_keypoints[i] = source_center + (source_keypoints[i] - source_center) * scale_factor

    # Calculate a uniform scale factor for the rest of the face based on the width and height
    source_width = np.linalg.norm(source_keypoints[width_indices[0]] - source_keypoints[width_indices[1]])
    target_width = np.linalg.norm(target_keypoints[width_indices[0]] - target_keypoints[width_indices[1]])
    width_scale = target_width / source_width if source_width != 0 else 1

    source_height = np.linalg.norm(source_keypoints[height_indices[0]] - source_keypoints[height_indices[1]])
    target_height = np.linalg.norm(target_keypoints[height_indices[0]] - target_keypoints[height_indices[1]])
    height_scale = target_height / source_height if source_height != 0 else 1

    # Apply uniform scaling to the rest of the keypoints
    for i in rest_indices:
        relative_position = source_keypoints[i] - source_center
        relative_position[0] *= width_scale  # Apply width scale
        relative_position[1] *= height_scale  # Apply height scale
        scaled_keypoints[i] = source_center + relative_position

    # Translate the scaled keypoints to align with the target's center
    translation_vector = target_center - source_center
    scaled_keypoints[outline_indices] += translation_vector
    scaled_keypoints[rest_indices] += translation_vector

    return scaled_keypoints



def draw(source, W, H, face=True):
    detected_map = draw_pose(source, H, W, face)
    detected_map = HWC3(detected_map)


    detected_map = cv2.resize(
        detected_map, (W, H), interpolation=cv2.INTER_LINEAR
    )
    detected_map = Image.fromarray(detected_map)
    return detected_map



def rescale_skeleton(source, targ):
    res = copy(source)
    # body
    source_body = np.array(source["bodies"]["candidate"])
    targ_body = np.array(targ["bodies"]["candidate"])
    for i in range(len(source_body)):
        if source["bodies"]["subset"][0][i] == -1:
            source_body[i] = np.array([-1., -1.])
        if targ["bodies"]["subset"][0][i] == -1:
            targ_body[i] = np.array([-1., -1.])
        
    res_body = rescale_body_keypoints(source_body, targ_body, pairs["bodies"], 1)
    res["bodies"]["candidate"] = res_body

    # hands
    hand = [True, True]
    source_hands = np.array(source["hands"])
    targ_hands = np.array(targ["hands"])
    res_hands = []
    for i in range(len(targ_hands)):
        if np.sum(np.all(source_hands[i]==np.array([-1., -1.]), axis=1)) >= 5:
            source_hands[i][:] = -1.
            hand[i] = False
        res_hands.append(rescale_body_keypoints(source_hands[i], targ_hands[i], pairs["hands"], 0))

    res["hands"] = np.array(res_hands)


    # face
    source_face = np.array(source["faces"][0])
    targ_face = np.array(targ["faces"][0])
    source_face_not_detected = np.sum(np.all(source_face==np.array([-1., -1.]), axis=1))
    targ_face_not_detected = np.sum(np.all(targ_face==np.array([-1., -1.]), axis=1))
    
    face_transform = False
    if source_face_not_detected < 20 and targ_face_not_detected < 20:
        # outline_indices = list(range(17))  
        # rest_indices = list(range(17, 68))  
        # width_indices = pairs["face ref indices"][0]
        # height_indices = pairs["face ref indices"][2]
        # res_face = apply_face_scaling(
        #     source_face,
        #     targ_face,
        #     outline_indices,
        #     rest_indices,
        #     width_indices,
        #     height_indices
        # )
        M, inliers = cv2.estimateAffinePartial2D(source_face, targ_face)

        res_face = cv2.transform(np.array([source_face]), M)[0]
        res["faces"][0] = np.array(res_face)
        
        face_transform = True


    # move face and hands
    # if face_transform:
    face_delta_x_y = res["faces"][0][30] - res["bodies"]["candidate"][0]
    res["faces"][0] = res["faces"][0] - face_delta_x_y

    if hand[1]:
        left_hand_delta_x_y = res["hands"][1][0] - res["bodies"]["candidate"][4]
        res["hands"][1] = res["hands"][1] - left_hand_delta_x_y

    if hand[0]:
        right_hand_delta_x_y = res["hands"][0][0] - res["bodies"]["candidate"][7]
        res["hands"][0] = res["hands"][0] - right_hand_delta_x_y
    
    return res