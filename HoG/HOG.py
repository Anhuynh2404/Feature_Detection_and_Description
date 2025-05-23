import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

CELL_SIZE =(8, 8)
TARGET_IMG_SIZE = (64, 128)
BLOCK_SIZE_CELLS = (2, 2)
N_BINS = 9
ANGLE_MAX = 180 

def load_and_preprocess_image(image_path, target_size=TARGET_IMG_SIZE):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((target_size[0], target_size[1]), Image.Resampling.LANCZOS)
        img_gray = img_resized.convert('L')
        return np.array(img_gray, dtype=float) / 255.0
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
def gradient(image):
    gradient_x = np.zeros_like(image, dtype=float)
    gradient_y = np.zeros_like(image, dtype=float)

    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            if j == 0:
                gradient_x[i,j] = image[i,j+1] - image[i,j]
            elif j == cols - 1:
                gradient_x[i,j] = image[i,j] - image[i,j-1]
            else:   
                gradient_x[i,j] = (image[i,j+1] - image[i,j-1]) / 2.0

    for i in range(rows):
        for j in range(cols):
            if i == 0:
                gradient_y[i,j] = image[i+1,j] - image[i,j]
            elif i == rows - 1:
                gradient_y[i,j] = image[i,j] - image[i-1,j]
            else:   
                gradient_y[i,j] = (image[i+1,j] - image[i-1,j]) / 2.0

    return gradient_x, gradient_y

def caculation_gradient_magnitude(gradient_y, gradient_x):
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return magnitude

def caculation_gradient_direction(gradient_y, gradient_x):
    direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    direction[direction < 0] += 180
    return direction

def calculate_gradient(img, template):
    ts = template.size #Number of elements in the template (3).
    #New padded array to hold the resultant gradient image.
    new_img = np.zeros((img.shape[0]+ts-1, 
                           img.shape[1]+ts-1))
    new_img[np.uint16((ts-1)/2.0):img.shape[0]+np.uint16((ts-1)/2.0), 
            np.uint16((ts-1)/2.0):img.shape[1]+np.uint16((ts-1)/2.0)] = img
    result = np.zeros((new_img.shape))
    
    for r in np.uint16(np.arange((ts-1)/2.0, img.shape[0]+(ts-1)/2.0)):
        for c in np.uint16(np.arange((ts-1)/2.0, 
                              img.shape[1]+(ts-1)/2.0)):
            curr_region = new_img[r-np.uint16((ts-1)/2.0):r+np.uint16((ts-1)/2.0)+1, 
                                  c-np.uint16((ts-1)/2.0):c+np.uint16((ts-1)/2.0)+1]
            curr_result = curr_region * template
            score = np.sum(curr_result)
            result[r, c] = score
    #Result of the same size as the original image after removing the padding.
    result_img = result[np.uint16((ts-1)/2.0):result.shape[0]-np.uint16((ts-1)/2.0), 
                        np.uint16((ts-1)/2.0):result.shape[1]-np.uint16((ts-1)/2.0)]
    return result_img

def gradient_magnitude(horizontal_gradient, vertical_gradient):
    horizontal_gradient_square = np.power(horizontal_gradient, 2)
    vertical_gradient_square = np.power(vertical_gradient, 2)
    sum_squares = horizontal_gradient_square + vertical_gradient_square
    grad_magnitude = np.sqrt(sum_squares)
    return grad_magnitude

def gradient_direction(horizontal_gradient, vertical_gradient):
    grad_direction = np.arctan(vertical_gradient/(horizontal_gradient+0.00000001))
    grad_direction = np.rad2deg(grad_direction)
    grad_direction = grad_direction%180
    return grad_direction