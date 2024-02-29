from monai.visualize import blend_images
import torch
import matplotlib.pyplot as plt


# https://github.com/Project-MONAI/tutorials/blob/main/experiment_management/spleen_segmentation_mlflow.ipynb
def render(image, label, prediction, show=False, out_file=None, colormap="spring"):
    """
    Render a two-column overlay, where the first column is the target (correct) label atop the original image,
    and the second column is the predicted label atop the original image.

    Args:
        image: the input image to blend with label and prediction data.
        label: the input label to blend with image data.
        prediction: the predicted label to blend with image data.
        show: whether the figure will be printed out. default to False.
        out_file: directory to save the output figure. if none, no save happens. default to None.
        colormap: desired colormap for the plot. default to `spring`. for more details, please refer to:
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    correct_blend = blend_images(image=image, label=label, alpha=0.5, cmap=colormap, rescale_arrays=False)
    predict_blend = blend_images(image=image, label=prediction, alpha=0.5, cmap=colormap, rescale_arrays=False)
    lower, rnge = 5, 5
    count = 1
    fig = plt.figure("blend image and label", (8, 4 * rnge))
    for i in range(lower, lower + rnge):
        # plot the slice 50 - 100 of image, label and blend result
        slice_index = 10 * i
        plt.subplot(rnge, 2, count)
        count += 1
        plt.title(f"correct label slice {slice_index}")
        plt.imshow(torch.moveaxis(correct_blend[:, :, :, slice_index], 0, -1))
        plt.subplot(rnge, 2, count)
        count += 1
        plt.title(f"predicted label slice {slice_index}")
        plt.imshow(torch.moveaxis(predict_blend[:, :, :, slice_index], 0, -1))
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()
    return fig