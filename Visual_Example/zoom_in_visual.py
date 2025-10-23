import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def create_zoomed_inset(image_path, zoom_bbox, inset_position, inset_size,
                        border_color_region='yellow', border_color_inset='red',
                        border_width=4):
    """
    Create an image with a zoomed inset overlay.

    Parameters:
    -----------
    image_path : str
        Path to the input image
    zoom_bbox : tuple
        (x, y, width, height) defining the region to zoom in on
    inset_position : tuple
        (x, y) top-left corner position for the inset in the output image
    inset_size : tuple
        (width, height) size of the inset box
    border_color : str
        Color of the border around zoom region and inset
    border_width : int
        Width of the border in pixels

    Returns:
    --------
    numpy.ndarray : The output image with inset
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Extract zoom region
    x, y, w, h = zoom_bbox
    zoom_region = img_array[y:y + h, x:x + w]

    # Resize zoom region to inset size
    zoom_pil = Image.fromarray(zoom_region)
    inset_w, inset_h = inset_size
    zoom_resized = zoom_pil.resize((inset_w, inset_h), Image.LANCZOS)
    zoom_resized_array = np.array(zoom_resized)

    # Create output image (copy of original)
    output = img_array.copy()

    # Add border to zoom region on original image
    color_map = {'red': [255, 0, 0], 'green': [0, 255, 0],
                 'blue': [0, 0, 255], 'yellow': [255, 255, 0]}
    border_rgb_region = color_map.get(border_color_region, [255, 0, 0])
    border_rgb_inset = color_map.get(border_color_inset, [255, 0, 0])

    # Draw border around zoom region
    for i in range(border_width):
        # Top and bottom
        output[y + i, x:x + w] = border_rgb_region
        output[y + h - 1 - i, x:x + w] = border_rgb_region
        # Left and right
        output[y:y + h, x + i] = border_rgb_region
        output[y:y + h, x + w - 1 - i] = border_rgb_region

    # Place inset with border
    inset_x, inset_y = inset_position

    # Draw border around inset
    for i in range(border_width):
        # Top and bottom
        output[inset_y + i, inset_x:inset_x + inset_w] = border_rgb_inset
        output[inset_y + inset_h - 1 - i, inset_x:inset_x + inset_w] = border_rgb_inset
        # Left and right
        output[inset_y:inset_y + inset_h, inset_x + i] = border_rgb_inset
        output[inset_y:inset_y + inset_h, inset_x + inset_w - 1 - i] = border_rgb_inset

    # Place the zoomed content
    output[inset_y + border_width:inset_y + inset_h - border_width,
    inset_x + border_width:inset_x + inset_w - border_width] = \
        zoom_resized_array[border_width:-border_width, border_width:-border_width]

    return output


# Example usage
if __name__ == "__main__":
    # Define your image paths
    image_paths = [
        '289_0_GT.png',
        '289_0_Proposed.png',
        '289_0_RadioDiff.png',
        '289_0_RadioUNet-F.png',
        '289_0_RadioUNet-L.png'
    ]

    # Define zoom parameters (adjust these for your images)
    zoom_bbox = (170, 5, 70, 70)  # (x, y, width, height) - region to zoom
    inset_position = (5, 120)  # Top-left corner of inset
    inset_size = (130, 130)  # Size of the inset box

    # Process all images
    results = []
    for img_path in image_paths:
        result = create_zoomed_inset(img_path, zoom_bbox, inset_position, inset_size)
        results.append(result)

    # Display all images in a row
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    titles = ['Ground Truth', 'Proposed', 'RadioDiff', 'RadioUNet-F', 'RadioUNet-L']

    for ax, img, title in zip(axes, results, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('super_resolution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Optionally save individual images
    for i, (img, path) in enumerate(zip(results, image_paths)):
        output_path = path.replace('.png', '_with_inset.png')
        Image.fromarray(img).save(output_path)
        print(f"Saved: {output_path}")