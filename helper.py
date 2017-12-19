import numpy as np
from PIL import Image

def plot_image(img):
    return Image.fromarray(img.astype(np.uint8))

# This function loads an image and returns it as a numpy array of floating-points.
# The image can be automatically resized so the largest of the height or width equals max_size.
# or resized to the given shape
def load_image(filename, shape=None, max_size=None):
    image = Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = float(max_size) / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, Image.LANCZOS) # PIL.Image.LANCZOS is one of resampling filter

    if shape is not None:
        image = image.resize(shape, Image.LANCZOS) # PIL.Image.LANCZOS is one of resampling filter

    # Convert to numpy floating-point array.
    return np.float32(image)