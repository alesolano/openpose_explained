#!/usr/bin/env python

from __future__ import division # division returns a floating point number
import os

import numpy as np
import cv2

from enum import Enum

class CocoPart(Enum):
    '''
    List of body parts' indices
    '''
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

class CocoPair(Enum):
    '''
    List of body pairs' indices
    '''
    LShoulder = 0
    RShoulder = 1
    RArm = 2
    RForearm = 3
    LArm = 4
    LForearm = 5
    RBody = 6
    RThigh = 7
    RCalf = 8
    LBody = 9
    LThigh = 10
    LCalf = 11
    Neck = 12
    RNoseEye = 13
    REyeEar = 14
    LNoseEye = 15
    LEyeEar = 16
    RShoulderEar = 17
    LShoulderEar = 18

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), #10
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17) # 19
]   # = 19
CocoPairsRender = CocoPairs[:-2]
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5), #9
    (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
 ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]



def rearrange_humans(humans):
    '''
    Transform the output of TfPoseEstimator's inference method.
    '''
    final_humans = []
    for human in humans:

        new_human = []
        
        for body_part in human.body_parts.values():
            new_part = {}
            new_part["idx"] = body_part.part_idx
            new_part["x_percent"] = body_part.x
            new_part["y_percent"] = body_part.y
            
            new_human.append(new_part)

        final_humans.append(new_human)

    return final_humans



class Humans():

    def __init__(self, humans, frame):
        self.image = frame
        self.image_h = frame.shape[0]
        self.image_w = frame.shape[1]
        
        self.humans = rearrange_humans(humans) # List of Human objects
        self.n_humans = len(self.humans) # Number of detected humans
        
        # Coordinates of all parts detected, ordered by humans.
        # List of humans. Each human is a dictionary of parts. Each part is a tuple of x and y coordinates.
        self.parts_coords = []
        
        # Vector components (magnitude and direction) of all pairs detected, ordered by humans
        # List of humans. Each human is a dictionary of pairs. Each pair is a tuple of magnitude and direction
        self.pairs_components = []
        
        if self.n_humans != 0:
            self.fill_pairs_components()
        

    def fill_pairs_components(self):
        '''
        Transform from the information provided by the HumanArray ROS message (detected body parts' positions relative to the dimensions of the camera, along with a value of confidence for each human) to three lists:
            - parts_coords: List of humans. Each human is a dictionary of parts. Each part is a tuple of x and y coordinates.
            - pairs_components: List of humans. Each human is a dictionary of pairs. Each pair is a tuple of magnitude and direction
            - certainties. List of confidence scores. Each score is associated with a human.
        '''    
        for human_idx, human in enumerate(self.humans):
            
            # Append an empty dictionary for each human detected.
            self.parts_coords.append({})
            self.pairs_components.append({})
            
            # For each detected part, transform from its relative position to the absolute position, knowing the dimensions of the image.
            # Fill the parts_coors dictionary associated with the human human_idx with this information.
            for part in human:
                x = int(part["x_percent"] * self.image_w + 0.5)
                y = int(part["y_percent"] * self.image_h + 0.5)
                self.parts_coords[human_idx][part["idx"]] = (x, y)

            # For each possible pair, get its magnitude and direction.
            # Fill the pairs_components dictionary associated with the human human_idx with this information.
            for pair_idx, pair in enumerate(CocoPairs):
                
                # If any of the parts that form the pair has not been detected, continue
                if pair[0] not in self.parts_coords[human_idx].keys() \
                    or pair[1] not in self.parts_coords[human_idx].keys():
                    continue

                x_0, y_0 = self.parts_coords[human_idx][pair[0]]
                x_1, y_1 = self.parts_coords[human_idx][pair[1]]

                magnitude = np.linalg.norm([x_1 - x_0, y_1 - y_0])
                direction = np.arctan2(y_1 - y_0, x_1 - x_0)
                self.pairs_components[human_idx][pair_idx] = (magnitude, direction)
                
    
    def draw(self, draw_position=False, draw_orientation=False):
        '''
        Combine all detected parts associated with a human to form colorful skeletons.
        '''
        image_drawn = np.copy(self.image)
        
        centers = {}
        for human_idx, human in enumerate(self.humans):
            
            # draw point
            for part_idx in range(len(CocoPart)):
                # if the part has not been detected, continue
                if part_idx not in self.parts_coords[human_idx].keys():
                    continue

                center = self.parts_coords[human_idx][part_idx]
                centers[part_idx] = center
                cv2.circle(image_drawn, center, 3, CocoColors[part_idx], thickness=3, lineType=8, shift=0)

            # draw line
            for pair_idx, pair in enumerate(CocoPairsRender):
                # if the pair has not been detected, continue
                if pair_idx not in self.pairs_components[human_idx].keys(): 
                    continue

                image_drawn = cv2.line(image_drawn, centers[pair[0]], centers[pair[1]],
                                       CocoColors[pair_idx], 3)

        return image_drawn