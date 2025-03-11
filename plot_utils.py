import matplotlib.pyplot as plt
import cv2

def show_image(image, title="Image"):
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
