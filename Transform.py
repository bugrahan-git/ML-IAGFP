import random
import cv2
import imgaug.augmenters as iaa
import numpy as np

"""Class to transform images with features random_rotation, random_noise, horizontal_flip"""


class Transform:

    def __init__(self):
        self.ctr = 0
        self.available_transformations = {
            'rotate': self.random_rotation,
            'horizontal_flip': self.horizontal_flip,
            'noise': self.add_noise,
            'crop': self.crop,
            'shear': self.shear,
        }

    def random_rotation(self, image_array: np.ndarray):
        rotate = iaa.Affine(rotate=(random.randint(-90, -1), random.randint(1, 179)))
        return rotate.augment_image(image_array)

    def add_noise(self, image_array: np.ndarray):
        gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
        return gaussian_noise.augment_image(image_array)

    def horizontal_flip(self, image_array: np.ndarray):
        flip_hr = iaa.Fliplr(p=1.0)
        return flip_hr.augment_image(image_array)

    def crop(self, image_array: np.ndarray):
        crop = iaa.Crop(percent=(0, 0.3))
        return crop.augment_image(image_array)

    def shear(self, image_array: np.ndarray):
        shear = iaa.Affine(shear=(0, 40))
        return shear.augment_image(image_array)

    def transform_image(self, image_to_transform, folder_path):
        num_transformations_to_apply = random.randint(1, len(self.available_transformations))

        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = random.choice(list(self.available_transformations))
            transformed_image = self.available_transformations[key](image_to_transform)
            num_transformations += 1

        new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, self.ctr)
        # write image to the disk
        cv2.imwrite(new_file_path, transformed_image)
        self.ctr += 1
        return transformed_image
