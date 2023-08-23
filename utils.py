"""
utils.py

Utility functions for the project.
"""
import os
import PIL
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL.ImageDraw import Draw
import seaborn as sns

dataset_path = 'dataset/frames/frames'


def get_file_path(
    image_id: int
) -> str:
    """
    Function reconstructs file path from image id.
    
    Parameters
    ----------
    image_id : int
        Image id.
    
    Returns
    -------
    str
        File path.
    """
    
    image_id = str(image_id).rjust(6, '0')
    return os.path.join(
        dataset_path,
        f'seq_{image_id}.jpg'
    )


def detect_objects_in_image(
    path: str,
    model: tf.keras.Model
) -> dict:
    """
    Extracts image from a file,
    adds new axis and passes the
    image through object detection model.
    
    Parameters
    ----------
    path : str
        File path.
    model : tf.keras.Model
        Object detection model.
    
    Returns
    -------
    dict
        Model output dictionary.
    """
    image_tensor = tf.image.decode_jpeg(
        tf.io.read_file(path),
        channels=3
    )[tf.newaxis, ...]
    return model(image_tensor)


def count_people_in_image(
    path: str,
    model: tf.keras.Model,
    threshold:float=0.
) -> int:
    """
    Counts the number of persons in an image.
    
    Parameters
    ----------
    path : str
        File path.
    model : tf.keras.Model
        Object detection model.
    threshold : float, optional
        Threshold for confidence scores, by default 0.
        
    Returns
    -------
    int
        Number of people for one image.
    """
    results = detect_objects_in_image(
        path,
        model
    )
    # Class ID 1 = "person"
    return (results['detection_classes'].numpy()[0] == 1)[np.where(
        results['detection_scores'].numpy()[0] > threshold)].sum()


def draw_boxes_on_image(
    image_path: str,
    data: dict,
    threshold:float=0.
) -> PIL.Image:
    """
    Draws bounding boxes for detected persons.
    
    Parameters
    ----------
    image_path : str
        File path.
    data : dict
        Model output dictionary.
    
    Returns
    -------
    PIL.Image
        Image with bounding boxes.
    """
    image = PIL.Image.open(image_path)
    draw = Draw(image)

    im_width, im_height = image.size

    boxes = data['detection_boxes'].numpy()[0]
    classes = data['detection_classes'].numpy()[0]
    scores = data['detection_scores'].numpy()[0]

    for i in range(int(data['num_detections'][0])):
        if classes[i] == 1 and scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = (
                xmin * im_width,
                xmax * im_width,
                ymin * im_height,
                ymax * im_height
            )
            draw.line(
                [
                    (left, top),
                    (left, bottom),
                    (right, bottom),
                    (right, top),
                    (left, top)
                ],
                width=4,
                fill='red'
            )

    return image


def set_style():
    """
    Sets display options for charts and pd.DataFrames.
    """

    # Plots display settings
    sns.set_style('darkgrid')
    plt.rcParams['figure.figsize'] = 12, 8
    plt.rcParams.update({'font.size': 14})
    # DataFrame display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.display.float_format = '{:.4f}'.format
    

def plot_distribution(
    dataset: pd.DataFrame,
    display: bool = False,
    save: bool = True
) -> None:
    """
    Plots distribution of people in images.
    
    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    b = np.histogram_bin_edges(dataset['count'], bins='sqrt')
    ax.hist(
        dataset['count'],
        bins = 25,
        edgecolor='black',
        linewidth=1.,
        color='red',
        width=1
    )
    
    ax.set_title('Distribution of people in images')
    ax.set_xlabel('Number of people')
    ax.set_ylabel('Number of images')

    if display:
        plt.show()
    if save:
        fig.savefig('plots/distribution.png', dpi=300)