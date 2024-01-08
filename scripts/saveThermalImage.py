import cv2
import numpy as np

def saveThermalImage(img_path):
    # Load the grayscale IR image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Normalize the image intensity values to the range [0, 1]
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Apply a colormap to simulate thermal colors
    thermal_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)

    # Display the thermal image
    cv2.imshow('Thermal Image', thermal_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the thermal image
    cv2.imwrite('thermal_'+ img_path, thermal_image)

saveThermalImage('test.png')