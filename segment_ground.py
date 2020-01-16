# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2


class SegmentationNN:
    def __init__(self):
        return

    def load_enet_model_from_file(self, model_filepath: str, class_filepath: str, class_color_filepath: str):
        self.enet_model = cv2.dnn.readNet(model_filepath)
        with open(class_filepath) as class_file:
            self.classes = class_file.readlines()
        with open(class_color_filepath) as color_file:
            colors = color_file.readlines()
            colors = [np.uint8(color.split(',')) for color in colors]
            self.class_colors = [np.array(color) for color in colors]
            self.class_colors = np.array(self.class_colors)

    def load_image_from_file(self, image_filepath: str):
        self.loaded_img = cv2.imread(image_filepath)
        self.loaded_img = cv2.resize(self.loaded_img, (1024, 512))
        img_blob = cv2.dnn.blobFromImage(
            self.loaded_img, 1 / 255.0, (1024, 512), 0, swapRB=True, crop=False)
        self.enet_model.setInput(img_blob)

    def infer(self):
        start_time = time.time()
        results = self.enet_model.forward()
        end_time = time.time()

        return (results, end_time-start_time)

    def infer_and_visualize(self, classes_of_interest=None):
                # TODO: Implement Classes of Interest
        if self.classes is None or self.class_colors is None:
            return

        (results, elapsed_time) = self.infer()

        # results[0] is a matrix with each entry storing the list of each pixel's class probabilities
        # pick the largest one for each to set as the class
        pixel_labels = np.argmax(results[0], axis=0)  # is height X width
        pixel_color_mask = self.class_colors[pixel_labels]  # fancy indexing

        visualization_img = cv2.addWeighted(
            np.uint8(pixel_color_mask), 0.4, np.uint8(self.loaded_img), 0.6, 0)
        cv2.imshow('Visualization', visualization_img)
        cv2.waitKey(2000)

        return (results, elapsed_time)


if __name__ == '__main__':
    segmenter = SegmentationNN()
    segmenter.load_enet_model_from_file('ground_segmentation_models/enet-model.net',
                                        'ground_segmentation_models/enet-classes.txt', 'ground_segmentation_models/enet-colors.txt')
    segmenter.load_image_from_file('test_images/example_01.png')
    segmenter.infer_and_visualize()
