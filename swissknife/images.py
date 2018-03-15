"""
Image processing utilities.
"""
import numpy as np
from .files import FilesStream


def compute_featurewise_mean_and_std(target_size, *folders, load_image=None):
    """Reads all images from folders and computes iterative estimation of
    variance and mean.

    Args:
        target_size: Resize all images from folder to this size.
        *folders: List of folders to be search for images.
        load_image: Function with the following signature:

                def load_image(filename, target_size):
                    pass

            Is called on each file and should  read image from disk and
            convert into Numpy array. If None, then default implementation is
            used.

    Returns:
         (mean, std): The tuple with dataset's mean and std values.

    """
    n = 1
    global_mean = None
    global_variance = np.ones(target_size)
    if load_image is None:
        load_image = FallbackImageLoader(channels_first=False)

    for folder in folders:
        stream = FilesStream(folder)
        for batch in stream(infinite=False):
            for image in batch:
                x = load_image(image, target_size=target_size)
                if global_mean is None:
                    global_mean = x
                else:
                    global_variance = (
                        (n - 2)*global_variance/(n - 1) +
                        (1./n)*(x - global_mean)**2)
                    global_mean = (1./n) * (x + (n - 1)*global_mean)
                n += 1

    return global_mean, np.sqrt(global_variance)


class FallbackImageLoader:
    """Class created from group of Keras utilities, providing fallback
    implementation to load images from filesystem into NumPy arrays
    if no any images reading libraries installed.

    This implementation is intentionally kept more simple then original one.
    """

    def __init__(self, channels_first=True, dtype=float):
        self.dtype = dtype
        self.channels_first = channels_first
        self.pil_image = None
        self.pil_interpolation = None
        self._import_pil()

    def _import_pil(self):
        """Makes an attempt to load PIL library and setup available
        interpolation methods.
        """
        try:
            from PIL import Image as pil_image
        except ImportError:
            raise ImportError(
                'Fallback image loader requires PIL library to '
                'convert image file into NumPy array.')
        else:
            self.pil_image = pil_image
            self.pil_interpolation = pil_image.NEAREST

    def load_image(self, path, target_size=None):
        """Loads an image into PIL format.

        Args:
            path: Path to image file
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.

        Returns:
            A numpy array.

        Raises:
            ImportError: If PIL is not available.
            ValueError: If interpolation method is not supported.

        """
        img = self.pil_image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                img = img.resize(width_height_tuple, self.pil_interpolation)
        return img

    def image_to_array(self, img):
        """Converts a PIL Image instance to a Numpy array.

        Args:
            img: PIL Image instance.

        Returns:
            A 3D Numpy array.

        Raises:
            ValueError: If invalid `img` or `data_format` is passed.

        """
        x = np.asarray(img, dtype=self.dtype)
        if len(x.shape) == 3:
            if self.channels_first:
                x = x.transpose(2, 0, 1)
        elif len(x.shape) == 2:
            if self.channels_first:
                x = x.reshape((1, x.shape[0], x.shape[1]))
            else:
                x = x.reshape((x.shape[0], x.shape[1], 1))
        else:
            raise ValueError('Unsupported image shape: ', x.shape)
        return x

    def __call__(self, filename, target_size=None):
        pil_image = self.load_image(filename, target_size=target_size)
        numpy_array = self.image_to_array(pil_image)
        return numpy_array
