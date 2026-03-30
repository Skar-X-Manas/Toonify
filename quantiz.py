import cv2
import numpy as np

def quantize_colors(image, k=8):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Convert to float32 for K-means
    pixels = np.float32(pixels)

    # Define criteria and apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to the centers to get the quantized image
    quantized_image = centers[labels.flatten()]

    # Reshape the quantized image to the original shape
    quantized_image = quantized_image.reshape(image.shape)

    return quantized_image

def cartoonize_image(image_path, output_path, k=8, resize_factor=0.6):
    # Read the image
    img = cv2.imread(image_path)

    # Resize the image
    height, width, _ = img.shape
    new_height = int(height * resize_factor)
    new_width = int(width * resize_factor)
    resized_img = cv2.resize(img, (new_width, new_height))

    # Apply bilateral filter for smoothing while preserving edges
    smoothed_image = cv2.bilateralFilter(resized_img, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    blurred_image = cv2.medianBlur(grayscale_image, 7)

    # Perform color quantization
    quantized_image = quantize_colors(smoothed_image, k)

    # Create an edge mask using adaptive thresholding
    edges = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    # Combine edges and quantized colors to create a cartoon effect
    cartoonized_image = cv2.bitwise_and(quantized_image, quantized_image, mask=edges)

    # Display the original and cartoonized images
    # cv2.imshow('Original Image', resized_img)
    cv2.imshow('Cartoonized Image', cartoonized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the cartoonized image
    cv2.imwrite(output_path, cartoonized_image)


cartoonize_image('ros.jpg', 'output_cartoon.jpg')
