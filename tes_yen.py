from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave

img = imread('images/Arjie.jpg')

yen_threshold = threshold_yen(img)
bright = rescale_intensity(img, (0, yen_threshold), (0, 255))

imsave('images/Arjie_yen.jpg', bright)
