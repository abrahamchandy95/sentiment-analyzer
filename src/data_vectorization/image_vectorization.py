import numpy as np
from typing import List, Tuple
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.transform import resize

class ImageVectorizer:
    def __init__(self, dims: Tuple[int, int]) -> None:
        self.dims = dims
    
    def convert_image_to_array(
        self, image_path: str
    ) -> np.ndarray:
        image_array = imread(image_path)
        if len(image_array.shape) == 2: #greyscale image
            image_array = gray2rgb(image_array)
        elif len(image_array.shape[2]) > 3:
            image_array = image_array[: , :, : 3]
        image_array = resize(
            image_array, (self.dims[0], self.dims[1]),
            anti_aliasing=True    
        )
        return image_array

    def extract_features_from_images(
    
        self, image_paths: List[str]
    ) -> np.ndarray:
        """Extracts features from a list of images.

        Args:
            image_paths (List[str]): A list of images 

        Returns:
            np.ndarray: Features of the images
        """
        num_images = len(image_paths)
        # initialize with zeros
        image_features = np.zeros(
            (num_images, self.dims[0], self.dims[1], 3),
            dtype=np.float32
        )
        for idx, img_path in enumerate(image_paths):
            image_features[idx] = self.convert_image_to_array(img_path)
        return image_features
    