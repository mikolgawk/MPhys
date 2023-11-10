from PIL import Image
import os

def divide_image(image_path, input_image_name,output_folder):
    # Open the image
    img = Image.open(image_path)

    # Convert the image to grayscale
    img = img.convert('L')

    # Get the original width and height
    original_width, original_height = img.size

    # Define the target size (96x96 pixels)
    target_size = (96, 96)

    # Calculate the number of 96x96 segments to extract horizontally
    num_horizontal_segments = original_width // target_size[0]

    for i in range(num_horizontal_segments):
        # Calculate the cropping region for the current segment
        left = i * target_size[0]
        upper = 0
        right = (i + 1) * target_size[0]
        lower = target_size[1]

        # Crop the segment from the original image
        segment = img.crop((left, upper, right, lower))

        # Save the segment as a new PNG file
        segment.save(f'{output_folder}/'+f'{input_image_name}'+f'_segment_{i}.png', 'PNG')

if __name__ == "__main__":
    start_index = 1
    end_index = 5
    for x in range(start_index, end_index + 1):

        input_image_directory = "C:/Users/Czaja/Desktop/MPhys thesis/Materials_project_picture_prep/venv/"
        input_image_name = 'band_structure_mp-' + str(x)
        input_image_path = input_image_directory + input_image_name + '.png'

        if not os.path.isfile(input_image_path):
            print("No such file %s" % input_image_path)
            continue

        output_folder = "96x96 divided pictures"  # Change this to the desired output folder

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        divide_image(input_image_path, input_image_name, output_folder)
