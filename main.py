import cv2
import numpy as np
from fastiecm import fastiecm

image = cv2.imread("habs_imgs/madagascar_habs.jpg")
# image = cv2.imread("park.png")
# image = cv2.imread("plants.png")
# image = cv2.imread("pasted image 0.png")

def display(win_name, image, scale=0.2, wait=True):    
    image = np.array(image, dtype=float)/float(255)
    shape = image.shape
    height = int(shape[0] * scale)
    width = int(shape[1] * scale)
    image = cv2.resize(image, (width, height))
    cv2.namedWindow(win_name)
    cv2.imshow(win_name, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def contrast_stretch(im, per=5):
    # find the top brightness of pixels in the image in 
    # the top 5% and bottom 5% of your image.
    in_min = np.percentile(im, per)
    # print(in_min)
    in_max = np.percentile(im, 100 - per)
    # print(in_max)
    # set the maximum brightness and minimum brightness on the 
    # new image you are going to create. The brightest a pixel’s 
    # colour can be is 255, and the lowest is 0
    out_min = 0.0
    out_max = 255.0

    # change all the pixels in the image, so that the image has the 
    # full range of contrasts from 0 to 255.
    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out
    
def calc_ndvi(image):
    '''
    Now that you have a high contrast image, it’s time to do the 
    NDVI calculations. This will take all the blue pixels and make 
    them brighter, and make all the red pixels darker, leaving an 
    image that will be black and white. The brightest pixels in 
    the image indicate healthy plants, and the darkest pixels 
    ndicate unhealthy plants or an absence of plants.
    '''
    # To adjust the pixels in the image and only work with red and blue, 
    # the image needs splitting into its three seperate channels. 
    # r for red, g for green, and b for blue.
    b, g, r = cv2.split(image)
    # Now the red and blue channels need to be added together and stored as bottom
    bottom = (r.astype(float) + b.astype(float))
    # Because we’re doing a division, we also need to make sure 
    # that none of our divisors are 0, or there will be an error
    bottom[bottom==0] = 0.01
    # The blue channel can then have the red channel subtracted 
    # (remember that red would mean unhealthy plants or no plants), 
    # and then divided by the bottom calculation
    ndvi = (b.astype(float) - r) / bottom
    # print(ndvi)
    print("avg NDVI:", -0.2*np.mean(ndvi))
    # To once again enhance the image, it can be run 
    # through the contrast_stretch function
    ndvi_contrasted = contrast_stretch(ndvi)
    return ndvi_contrasted

def colored_ndvi(image):
    # You can run the image through a colour mapping process that will 
    # turn really bright pixels to the colour red and dark pixels to the colour blue
    color_mapped_prep = image.astype(np.uint8)
    # The current image, that you have saved as ndvi_contrasted is not suitable 
    # for colour mapping. The numbers stored in the numpy array are currently 
    # all floats or what is commonly known as decimal numbers. They all need 
    # converting to whole numbers, or integers between 0 and 255
    color_mapped_image = cv2.applyColorMap(color_mapped_prep, fastiecm)

    return color_mapped_image

scale = 1
# display("Original", image, scale=scale, wait=False)
contrasted = contrast_stretch(image, per=5)
# display("Contrasted", contrasted, scale=scale, wait=False)
gray_ndvi = calc_ndvi(contrasted)
# print(gray_ndvi)
# display("GRAY NDVI", gray_ndvi, scale=scale, wait=False)
col_ndvi = colored_ndvi(gray_ndvi)
# print(col_ndvi)
display("COLORED NDVI", col_ndvi, scale=scale, wait=True)