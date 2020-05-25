from nnga.callbacks.visualization.classification_vis import classification_vis


def make_custom_visualization_by_task(task):
    visualizations = {
        "Classification": classification_vis,
    }

    if task not in visualizations is None:
        raise RuntimeError(
            "There isn't a valid task configured!\n \
                            Check your experiment config"
        )

    return visualizations.get(task)
