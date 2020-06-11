from nnga.callbacks.visualization.classification_vis import classification_vis
from nnga.callbacks.visualization.segmentation_vis import segmentation_vis


def make_custom_visualization_by_task(task):
    visualizations = {
        "Classification": classification_vis,
        "Segmentation": segmentation_vis,
    }

    return visualizations.get(task)
