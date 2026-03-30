import cv2
import numpy as np

def cartoonize_image(image_path, output_path, resize_factor=0.2):
    # Read the image
    img = cv2.imread(image_path)

    # Resize the image
    height, width, _ = img.shape
    new_height = int(height * resize_factor)
    new_width = int(width * resize_factor)
    resized_img = cv2.resize(img, (new_width, new_height))

    # Convert the resized image to grayscale
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Apply a bilateral filter to reduce noise while keeping edges sharp
    smooth = cv2.bilateralFilter(gray, 9, 300, 300)

    # Create an edge mask using adaptive thresholding
    edges = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    # Combine the smoothed image and the edge mask to create a cartoon effect
    cartoon = cv2.bitwise_and(resized_img, resized_img, mask=edges)

    # Display the original and cartoonized images
    #cv2.imshow('Original Image', resized_img)
    cv2.imshow('Cartoonized Image', cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the cartoonized image
    cv2.imwrite(output_path, cartoon)

# Replace 'path/to/your/image.jpg' with the actual path to your image file
# Replace 'path/to/your/output_cartoon.jpg' with the desired output path for the cartoonized image
cartoonize_image('isha.jpg', 'output_cartoon.jpg')
