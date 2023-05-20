import os
import sys

from skimage import io, color, filters, morphology, measure

# Get the input file name from the command line arguments
file = sys.argv[1] if len(sys.argv) == 2 else None

# Check the input file
if file is None or not os.path.isfile(file):
    print('$ python program.py <input_file>')
    print('Please enter a valid file path!')
    exit(1)

# Read the input image file
image_input = io.imread(file)

# Find a threshold for the image to convert it into binary
threshold = filters.threshold_otsu(image_input)

# Convert the gray image into binary image
binary_image = image_input < threshold

# Create a structuring element for morphology process
kernel = morphology.rectangle(15, 80)

# Perform dilation on the binary image
dilated_image = morphology.dilation(binary_image, kernel)

# Perform erosion on the dilated image
eroded_image = morphology.erosion(dilated_image, kernel)

# Find connected regions in the image
labeled_image = morphology.label(eroded_image)

# Find the coordinates for each region
regions = measure.regionprops(labeled_image)

# Create a rgb copy of the input image
output_image = color.gray2rgb(image_input)

# Demonstrate the detected regions on the image with red rectangles
for region in regions:
    height, width = kernel.shape
    min_row, min_col, max_row, max_col = region.bbox
    # Ignore regions that are smaller than the kernel
    if max_row - min_row < height or max_col - min_col < width:
        continue
    red = [255, 0, 0]
    output_image[min_row:max_row, min_col] = red
    output_image[min_row:max_row, max_col - 1] = red
    output_image[min_row, min_col:max_col] = red
    output_image[max_row - 1, min_col:max_col] = red

# Save the output image
io.imsave('output.png', output_image)
