import cv2
from matplotlib import pyplot as plt

sigmas = [0.5, 2.5, 3.1095, 4.1111, 5.9994, 6.3443]
plt.figure(figsize=(15, 10))

for i, sigma in enumerate(sigmas):
    # Construct image path - adjust this pattern to match your actual image names
    img_path = f'./img/{i + 3}.png'  # or whatever your naming pattern is

    # Read the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

    # Create subplot
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f'σ={sigma:.4f}')
    plt.axis('off')

plt.tight_layout()
plt.show()