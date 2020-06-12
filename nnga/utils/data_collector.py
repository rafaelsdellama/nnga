import os
from pathlib import Path


def read_dataset(main_folder, format="png"):
    """
    Read an entire dataset folder using a defined set of image formats.

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

        Therefore, the classes (labels) will be defined by the folder name that
        contains its image examples.

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
            dataset["labels"].append(__infer_label(path))
            dataset["data"].append(path)
        except (ValueError, FileExistsError):
            dataset["errors"].append(path)
    return dataset


def __infer_label(filepath):
    dir_hierarchy = str(Path(filepath).parents[0])
    label = dir_hierarchy.split(os.sep)[-1]

    return label


def collect_filepaths(main_path, format="png"):
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
