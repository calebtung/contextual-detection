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
            self.classes = np.array(class_file.read().splitlines())
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

    def infer_and_visualize(self, class_ids_of_interest=None):
        class_ids_of_interest = np.array(class_ids_of_interest)
        if self.classes is None or self.class_colors is None:
            return

        (results, elapsed_time) = self.infer()

        colors_of_interest = self.class_colors.copy()

        if class_ids_of_interest is not None:
            mask = np.ones(len(self.class_colors), np.bool)
            mask[class_ids_of_interest] = 0
            colors_of_interest[mask] = np.array([0,0,0])

        # results[0] is a matrix with each entry storing the list of each pixel's class probabilities
        # pick the largest one for each to set as the class
        pixel_labels = np.argmax(results[0], axis=0)  # is height X width
        pixel_color_mask = colors_of_interest[pixel_labels]  # fancy indexing

        # initialize the legend visualization
        legend = np.zeros(((len(class_ids_of_interest) * 25) + 25, 300, 3), dtype="uint8")
        
        # loop over the class names + colors
        for (i, (className, color)) in enumerate(zip(self.classes[class_ids_of_interest], self.class_colors[class_ids_of_interest])):
            # draw the class name + color on the legend
            color = [int(c) for c in color]
            cv2.putText(legend, className, (5, (i * 25) + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
                tuple(color), -1)

        cv2.imshow('Legend', legend)

        visualization_img = cv2.addWeighted(
            np.uint8(pixel_color_mask), 0.4, np.uint8(self.loaded_img), 0.6, 0)
        cv2.imshow('Visualization', visualization_img)
        cv2.waitKey(10000)

        return (results, elapsed_time)


if __name__ == '__main__':
    segmenter = SegmentationNN()
    segmenter.load_enet_model_from_file('ground_segmentation_models/enet-model.net',
                                        'ground_segmentation_models/enet-classes.txt', 'ground_segmentation_models/enet-colors.txt')
    segmenter.load_image_from_file('test_images/durres_street_view.jpg')
    segmenter.infer_and_visualize((1,2))
