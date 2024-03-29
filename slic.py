"""
A simple slic implementation for segmenting an image into superpixels.
"""

import numpy as np
import cv2
from skimage.color import rgb2lab
from skimage.segmentation.slic_superpixels import slic


def my_slic(
    image: np.ndarray,
    num_superpixels: int = 100,
    max_iter: int = 300,
    compactness: int = 10,
    convergence_threshold: float = 1.0,
):
    """
    Segment an image into superpixels using the SLIC algorithm.

    :param image: The input image.
    :param num_superpixels: The number of superpixels to generate.
    :param max_iter: The maximum number of iterations to run the algorithm.
    :param compactness: The compactness parameter.
    :param convergence_threshold: The threshold for cluster center change to determine convergence.
    :return: An array of labels, where each pixel is assigned a superpixel label.
    """

    # Convert the image to the LAB color space
    lab_image = rgb2lab(image)

    # Calculate the step size for evenly spacing the superpixels
    step = np.sqrt(image.shape[0] * image.shape[1] / num_superpixels)
    x_steps = np.linspace(
        0, image.shape[0] - 1, int(np.sqrt(num_superpixels)), dtype=int
    )
    y_steps = np.linspace(
        0, image.shape[1] - 1, int(np.sqrt(num_superpixels)), dtype=int
    )

    # Initialize the cluster centers
    cluster_centers = []
    for x in x_steps:
        for y in y_steps:
            cluster_center = [x, y] + list(lab_image[x, y])
            cluster_centers.append(cluster_center)
    cluster_centers = np.array(cluster_centers)

    # Initialize the labels and distances
    labels = -1 * np.ones(image.shape[:2], dtype=int)
    distances = np.full(image.shape[:2], np.inf)

    # Run the algorithm for a maximum of max_iter iterations
    for iteration in range(max_iter):
        # Store old cluster centers to check convergence later
        old_cluster_centers = np.copy(cluster_centers)

        for i, center in enumerate(cluster_centers):
            x, y, l, a, b = map(int, center)
            x_min, x_max = max(x - int(step), 0), min(x + int(step), image.shape[0] - 1)
            y_min, y_max = max(y - int(step), 0), min(y + int(step), image.shape[1] - 1)

            for j in range(x_min, x_max + 1):
                for k in range(y_min, y_max + 1):
                    d = np.linalg.norm([j - x, k - y])
                    color_distance = np.linalg.norm(lab_image[j, k] - center[2:])
                    distance = color_distance + (d / step) * compactness
                    if distance < distances[j, k]:
                        distances[j, k] = distance
                        labels[j, k] = i

        # Update cluster centers
        for i in range(len(cluster_centers)):
            mask = labels == i
            if np.any(mask):
                cluster_centers[i, :2] = np.mean(np.argwhere(mask), axis=0)
                cluster_centers[i, 2:] = np.mean(lab_image[mask], axis=0)

        # Check for convergence
        center_shift = np.linalg.norm(old_cluster_centers - cluster_centers, axis=1)
        if np.max(center_shift) < convergence_threshold:
            print(f"Converged at iteration {iteration}")
            break

    return labels


def draw_superpixel_boundaries(image: np.ndarray, labels: np.ndarray):
    edges = np.zeros_like(labels, dtype=bool)
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]  # Horizontal edge mask
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]  # Vertical edge mask
    outlined_image = image.copy()
    outlined_image[edges, :] = [255, 0, 0]  # Red color for boundaries
    return outlined_image


image = cv2.cvtColor(cv2.imread("brandeis.jpeg"), cv2.COLOR_BGR2RGB)

# My implementation
for compactness in [10, 30, 50]:
    labels = my_slic(image, num_superpixels=100, compactness=compactness, max_iter=20)
    outlined_image = draw_superpixel_boundaries(image, labels)
    cv2.imwrite(
        f"superpixels_{compactness}.png",
        cv2.cvtColor(outlined_image, cv2.COLOR_RGB2BGR),
    )

    # skimage implementation
    labels = slic(
        image,
        n_segments=100,
        compactness=compactness,
        max_num_iter=20,
        enforce_connectivity=False,
    )
    outlined_image = draw_superpixel_boundaries(image, labels)
    cv2.imwrite(
        f"superpixels_skimage_{compactness}.png",
        cv2.cvtColor(outlined_image, cv2.COLOR_RGB2BGR),
    )
    labels = slic(
        image,
        n_segments=100,
        compactness=compactness,
        max_num_iter=20,
        enforce_connectivity=True,
    )
    outlined_image = draw_superpixel_boundaries(image, labels)
    cv2.imwrite(
        f"superpixels_skimage_{compactness}_connected.png",
        cv2.cvtColor(outlined_image, cv2.COLOR_RGB2BGR),
    )
