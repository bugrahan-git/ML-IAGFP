import cv2

from src.Util.Transform import Transform

"""
    Python file to test code snippets
"""

image = cv2.imread("dataset/popart/roy-lichtenstein/0001.jpg")

t = Transform()

image = t.shear(image)

cv2.imshow("Rotated", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

