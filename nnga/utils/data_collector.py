import os
from pathlib import Path


def read_dataset(main_folder: str, format: str = "png", labeling: str = None):
    """
    Read an entire dataset folder using a defined set of image formats.

    Notes
    -----

    Take in consideration that not all the image formats are able to read using
    the `load_image` method called from `baseimgproc.io` library. In general,
    well known image format such as `.png`, `.tif` and `.jpg` are availble. The
    complete list of image formats can be found at baseimgproc.io method
    documentation.

    Parameters
    ----------
    main_folder : str
        The full path for a folder that contains image files to be defined as
        the model training dataset
    format : str, optional
        The file format that will be used to search the images in such input
        folder. This string can be a set of formats separated by comma, e.g.

        >>> format_set = "png, jpg, tif"

        By default "png" is adopted.
    labeling: str, optional
        The label that defines what class a loaded image belongs to is
        collected by two strategies: using a organized folder subtree
        architeture or suffix naming. The first option will be used when the
        `labeling` variable is None. In this situation, the user must had
        prepared the main_folder hierarchy with the following pattern:

            --> main_folder:
              --> cat/ [image list with several examples of cats pictures]
              --> dog/ [image list with several examples of dogs pictures]
              --> ant/ [image list with several examples of ants pictures]

        Therefore, the classes (labels) will be defined by the folder name that
        contains its image examples.

        The second option assumes that the classes (labels) information is
        provided by a suffix in the image name. For instance, considering that
        a cat image name is `cat_example_LABEL_1.png`, thus the `labeling`
        pattern is `_LABEL_`, which will extract the substring `1` as the label
        that defines the class `cat`.

    Raises
    ------
    FileExistsError
        Folder not found with provided path

    Returns
    -------
    dict
        A structured dataset informing the numpy arrays of images that were
        indeed read (by the key `images`), the list of image classes (by the
        key `labels`) and the list of images that encounter reading errors
        (by the key `errors`)
    """
    if not os.path.exists(main_folder):
        raise FileNotFoundError("Folder not found with provided path.")

    paths = collect_filepaths(main_folder, format)
    dataset = {"data": [], "labels": [], "errors": []}

    for path in paths:
        try:
            dataset["labels"].append(__infer_label(path, labeling))
            dataset["data"].append(path)
        except (ValueError, FileExistsError):
            dataset["errors"].append(path)
    return dataset


def __infer_label(filepath: str, labeling: str):
    label = ""
    if isinstance(labeling, str):
        filename = Path(filepath).stem
        label = filename.split(labeling)[-1]
    else:
        dir_hierarchy = str(Path(filepath).parents[0])
        label = dir_hierarchy.split(os.sep)[-1]

    return label


def collect_filepaths(main_path: str, format: str = "png"):
    """
    Collect a set of files using a determined location on disk and a known list
    of file formats. This is useful to get the full path of files encountered
    on a disk folder.

    Parameters
    ----------
    main_path : str
        The full path to the main folder where the many files are found in it.
        A recursive path collection is assumed, i.e. the files are found inside
        nested folder hierarchy.
    format : str, optional
        The file format that will be used to search the images in such input
        folder. This string can be a set of formats separated by comma, e.g.

        >>> format_set = "png, jpg, tif"

        By default "png" is adopted.

    Returns
    -------
    list
        A list with file paths collect using the specific file formats.
    """
    root_path = Path(main_path)
    format_list = format.split(",")
    paths = []
    for f in format_list:
        paths.extend(root_path.rglob(f"*.{f.strip()}"))

    return [path.as_posix() for path in paths]
