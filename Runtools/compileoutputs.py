import os
from PIL import Image

pdf_page_width = 650
savefolder = "outputs"

# These functions can be changed later to send all of the output to one place.
# For now, they are just used to compile the outputs for a single run into a single document for graphs and one for text.


# NOTE: OUTPUT GRAPH outputs/run_id.png WILL BE OVERWRITTEN EVERY TIME THIS FUNCTION IS CALLED
def compile_run_plots(run_id):
    """
    This function compiles the plots for a run into a single document.

    Args:
        run_id (str): The id of the run to compile the plots for.
        save_folder (str): The folder to save the compiled plots to.
    """
    image_paths = [run_id + "dataplot.png", "validation_loss.png", "classhist.png", "roc.png", "confusion_matrix.png"]

    images = [Image.open(image_path) for image_path in image_paths]

    # Get aspect ratios for images
    aspect_ratios = [image.width / image.height for image in images]
    # Get new heights for images based on aspect ratios and pdf_page_width
    # First image is full width, the rest are half width because they are smaller and therefore paired
    heights = [int(pdf_page_width / aspect_ratios[i]) if i == 0 else int((pdf_page_width / 2) / aspect_ratios[i]) for i in range(len(aspect_ratios))]

    # Resize images
    resized_images = [image.resize((pdf_page_width, heights[i])) if i == 0 else image.resize((int(pdf_page_width / 2), heights[i])) for i, image in enumerate(images)]

    # Create new image
    new_image = Image.new("RGB", (pdf_page_width, sum(heights[0:3])))

    # Paste images into new image
    x_offset = 0
    y_offset = 0
    # Use offsets to paste images into correct position, first image is full width, the rest are half width
    for i in range(len(resized_images)):
        new_image.paste(resized_images[i], (x_offset, y_offset))
        if i % 2 == 0:
            y_offset += resized_images[i].height
            x_offset = 0
        else:
            x_offset += resized_images[i].width

    # Save new image as run_id.png
    new_image.save(os.path.join(savefolder, run_id + ".png"))

    # Remove old images (individual graph pngs)
    for image_path in image_paths:
        os.remove(image_path)

    # Could convert to pdf but it blurs the images a bit
    # convert_png_to_pdf(run_id)



def convert_png_to_pdf(run_id):
    """
    This function converts a png to a pdf.

    Args:
        run_id (str): The id of the run to convert to a pdf.
    """
    image = Image.open(os.path.join(savefolder, run_id + ".png"))
    image.save(os.path.join(savefolder, run_id + ".pdf"))
    os.remove(os.path.join(savefolder, run_id + ".png"))



def get_output_text(run_id, params):
    """
    This function moves the text from the results.txt file for a run to outputs/run_id.txt.
    It also adds information about parameters.

    Args:
        run_id (str): The id of the run to get the text for.

    Returns:
        str: The text from the results.txt file for the run.
    """
    with open("results.txt", "r") as f:
        with open(os.path.join(savefolder, run_id + ".txt"), "w") as f2:
            f2.write(f"PARAMETERS\n\n")
            f2.write(f"training_feature_keys: {params.training_feature_keys}\n")
            f2.write(f"batch_size: {params.batch_size}\n")
            f2.write(f"n_epochs: {params.n_epochs}\n")
            f2.write(f"use_pca: {params.use_pca}\n")
            f2.write(f"train_data_size: {params.train_data_size}\n")
            f2.write(f"test_data_size: {params.test_data_size}\n")
            f2.write(f"valid_data_size: {params.valid_data_size}\n")
            f2.write(f"total_datasize: {params.total_datasize}\n")
            f2.write(f"is_local_simulator: {params.is_local_simulator}\n")
            f2.write(f"n_qubits: {params.n_qubits}\n")
            f2.write(f"num_layers: {params.num_layers}\n")
            f2.write(f"obs: {params.obs}\n")
            f2.write(f"spsa_alpha: {params.spsa_alpha}\n")
            f2.write(f"spsa_gamma: {params.spsa_gamma}\n")
            f2.write(f"spsa_c: {params.spsa_c}\n")
            f2.write(f"spsa_A: {params.spsa_A}\n")
            f2.write(f"spsa_a: {params.spsa_a}\n")
            f2.write(f"\n\nRESULTS\n\n")
            f2.write(f.read())
    
    os.remove("results.txt")