import math
import cv2
import random as rd
import matplotlib.pyplot as plt
from nnga.callbacks.visualization.commons import plot_to_image


def segmentation_vis(model, validation_dataset):
    # Use the model to predict the values from the test_images.
    idx = rd.randint(0, len(validation_dataset) - 1)
    batch_imgs, batch_masks = validation_dataset[idx]
    pred_masks = model.predict(batch_imgs)

    return {"Segmentation": plot_to_image(
        plot_masks(batch_imgs, batch_masks, pred_masks))
    }


def plot_masks(imgs, masks, predcit_masks):
    columns = 3
    rows = math.ceil(len(imgs))
    img_size = 2.5
    figure = plt.figure(figsize=(math.ceil(columns * img_size),
                                 math.ceil(rows * img_size)))

    for i in range(len(imgs)):
        # Start next subplot.
        predcit_mask = predcit_masks[i]
        mask = masks[i]
        img = imgs[i]

        if img.shape[2] == 1:
            img = img[:, :, 0]

        plt.subplot(rows, columns, i * 3 + 1, title="Image")
        plt.imshow(img, cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        mask = mask.astype('uint8')
        masked_data = cv2.bitwise_and(img, img, mask=mask)

        plt.subplot(rows, columns, i * 3 + 2, title="Mask")
        plt.imshow(masked_data, cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        _, predcit_mask = cv2.threshold(predcit_mask, 0.5, 255, cv2.THRESH_BINARY)
        predcit_mask = predcit_mask.astype('uint8')
        predict_masked_data = cv2.bitwise_and(img, img, mask=predcit_mask)

        plt.subplot(rows, columns, i * 3 + 3, title="Predict Mask")
        plt.imshow(predict_masked_data, cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

    figure.tight_layout()
    return figure
