# -*- coding: utf-8 -*-
"""
Based on : https://huggingface.co/google/owlvit-base-patch32
"""

import fitz
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import OwlViTProcessor, OwlViTForObjectDetection


def pdf_to_jpg(input_pdf_path: str, request_list: list, output_path: str,
               threshold: float = .15, sanity_check: bool = True,
               verbose: bool = True):
    '''
    Extracts crops of the items in request_list from an image or pdf.

    Parameters
    ----------
    input_pdf_path : str
        Path to input pdf.
    request_list : list
        List of descriptions of objects to be extracted.
    output_path : str
        Path to output folder for jpg.
    threshold : float, optional
        Minimum confidence for object detection, between [0,1].
        The default is .15.
    sanity_check : bool, optional
        Ouputs location of objects in mathplotlib. The default is True.
    verbose : bool, optional
        Add prints with confidence for each detected object.
        The default is True.

    Returns
    -------
    None.

    '''

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32")

    if img.endswith('.pdf'):
        doc = fitz.open(img)  # open document
        image_list = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap()
            imga = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            image_list.append(imga)
    else:
        image_list = [Image.open(img)]

    for image in image_list:
        inputs = processor(text=texts, images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes)

        if sanity_check:

            fig, ax = plt.subplots(1)
            plt.imshow(image)

        for i in range(len(request_list)):

            text = texts[i]
            boxes, scores, labels = (results[i]["boxes"], results[i]["scores"],
                                     results[i]["labels"])

            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                if verbose:
                    print("Detected "+str(text[label])+" with confidence "
                          + str(round(score.item(), 3))+" at location " +
                          str(box))

                x_b_left, y_b_left, x_t_right, y_t_right = box
                width = x_t_right - x_b_left
                height = y_t_right - y_b_left

                if text[label] == texts[0][0]:
                    rect = patches.Rectangle((x_b_left, y_b_left), width+15,
                                             height+15, linewidth=2,
                                             edgecolor='b', facecolor='none')
                else:
                    rect = patches.Rectangle((x_b_left, y_b_left), width+15,
                                             height+15, linewidth=2,
                                             edgecolor='r', facecolor='none')

                if sanity_check:
                    ax.add_patch(rect)

                if output_path is not None:
                    crop = image.crop((x_b_left, y_b_left,
                                       x_t_right, y_t_right))
                    crop.save(output_path+str(text[label])
                              + str(box)+'.jpeg', 'jpeg')
                    

    


if __name__ == '__main__':

    img = 'C:/Documents/Université/Master/Mémoire/Tests_cognitif/TAU-213.pdf'
    texts = [['a drawing of a cube']]

    output_path = 'C:/Documents/Université/Master/Mémoire/Output_horloges_patients/'

    pdf_to_jpg(img, texts, output_path, threshold=0.05)
    
    

