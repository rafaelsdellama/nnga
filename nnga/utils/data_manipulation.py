import cv2
import numpy as np


def scale_features(sample, header, scale_parameters, scale_method):
    """
        Parameters
        ----------
        sample : list
            List containing all to features to be scaled

        header : list
            Features name

        scale_parameters : dict
            Dict containing the scale parameters
                - Mean
                - Stdev
                - Min
                - Max

        scale_method : str
            Method to be used to scale the data
                - Standard
                - MinMax

        Returns
        -------
            sample scaled

    """
    scaled = []
    if scale_method == "Standard":
        for i, key in enumerate(header):
            mean = scale_parameters[key]["mean"]
            stdev = (
                scale_parameters[key]["stdev"]
                if scale_parameters[key]["stdev"] != 0
                else 0.001
            )
            scaled.append((sample[i] - mean) / stdev)

    elif scale_method == "MinMax":
        for i, key in enumerate(header):
            min_val = scale_parameters[key]["min"]
            max_val = scale_parameters[key]["max"]
            diff = (max_val - min_val) if max_val - min_val != 0 else 0.001
            scaled.append((sample[i] - min_val) / diff)

    else:
        return sample

    return scaled


def normalize_image(img):
    """
        Scale image

        Parameters
        ----------
        img : numpy array image
            img to be scaled

        Returns
        -------
        numpy array image scaled

    """
    return img / 255


def adjust_image_shape(img, input_shape, preserve_ratio=False):
    """
        Adjust image shape

        Parameters
        ----------
        img : numpy array image
            img to be scaled

        input_shape: tuple
            Tuple of image shape

        preserve_ratio: Bool
            if True, preserve image ratio,
            else Does not preserve the image ratio

        Returns
        -------
            numpy array image

        """

    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise ValueError(
            "input_shape is invalid. Should be (width, height, n_channels)"
        )

    if input_shape[2] == 1 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif input_shape[2] == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if preserve_ratio:
        scale_height = input_shape[0] / img.shape[0]
        scale_width = input_shape[1] / img.shape[1]
        scale_percent = (
            scale_height if scale_height < scale_width else scale_width
        )
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

        border_height = input_shape[0] - resized.shape[0]
        border_top = int((input_shape[0] - resized.shape[0]) / 2)
        border_bottom = border_height - border_top
        border_width = input_shape[1] - resized.shape[1]
        border_left = int((input_shape[1] - resized.shape[1]) / 2)
        border_right = border_width - border_left

        final_img = cv2.copyMakeBorder(
            resized,
            border_top,
            border_bottom,
            border_left,
            border_right,
            cv2.BORDER_CONSTANT,
        )
    else:
        dim = (input_shape[1], input_shape[0])
        final_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return np.reshape(final_img, input_shape)
