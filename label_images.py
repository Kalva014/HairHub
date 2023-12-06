import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from roipoly import RoiPoly

# Save labeled region of interest image to a file. Remove extension of file and convert from png to csv
def create_labeled_file(roi_data, image_file_name, image_path):
    image_file_name = image_file_name[0:-3] + 'csv'
    image_file_path = os.path.join(image_path, image_file_name)
    image_file = open(image_file_path, "x")
    np.savetxt(image_file_path, roi_data)
    image_file.close()

# Setup figure to label hair region of interest
def label_image(folder_path, image_to_label):
    unlabeled_image_path = os.path.join(folder_path, image_to_label)
    img = mpimg.imread(unlabeled_image_path)

    # Show the image
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.colorbar()
    plt.title("left click: line segment         right click or double click: close region")
    plt.show(block=False)

    # Let user draw first ROI
    hair_region_of_interest = RoiPoly(color='r', fig=fig)

    # Show ROI masks
    hair_mask = hair_region_of_interest.get_mask(img[:, :, 0])
    extra_mask = ~hair_mask

    plt.imshow(hair_mask,
            interpolation='nearest', cmap="Greys")
    plt.title('ROI masks of the two ROIs')
    plt.show()
    
    # hair_roi_pixels = img[hair_mask[:, :, 0]] # This gets pixels of region of interest
    # extra_roi_pixels = img[extra_mask]
    print(f"img shape: {img.shape}")
    print(f"mask shape: {hair_mask.shape}")
    create_labeled_file(hair_mask, image_to_label, os.path.join(os.getcwd(), 'data/masks/medium_hairstyles'))
    # create_labeled_file(extra_roi_pixels, image_to_label, os.path.join(os.getcwd(), 'data/Extra_Labeled/long_hairstyles'))

# Main Code Loop
def main():
    folder = os.path.join(os.getcwd(), 'data/initial_imgs/medium_hairstyles')
    for filename in os.listdir(folder):
        image_file_name = filename[0:-3] + 'csv'
        print(image_file_name)
        if image_file_name not in os.listdir('data/masks/medium_hairstyles'): # Check if file is already labeled
            label_image(folder, filename)

if __name__ == '__main__':
    main()
