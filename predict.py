import torch
from torchvision import datasets, models, transforms
import numpy as np

import argparse

from model.data import get_data
from model.model import load_model

import cv2
import os

import streamlit as st


import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--num_patches", type=int, default=9, help="split test image into x parts"
)
parser.add_argument(
    "--ckpt",
    type=str,
    default=r"C:\Users\z004fwwh\Desktop\thesis_code\thesis\experiments\test-time-CNN\resnet18_best_model_patch.pt",
    help="checkpoint folder to load model",
)
parser.add_argument("--model", type=str, default="resnet18", help="model to use")
parser.add_argument("--num_classes", type=int, default=7, help="number of classes")

args = parser.parse_args()

model_ft = load_model(args.model, args.ckpt, args.num_classes)

import math


def classify_batch(predictions):
    if len(set(predictions)) == 1:
        return predictions[0]
    elif 6 in predictions:
        predictions = list(filter(lambda item: item != 6, predictions))

        return max(set(predictions), key=predictions.count)
    else:
        return max(set(predictions), key=predictions.count)


def label_to_class(label):
    # create a dictionary of labels to class
    label_to_class = {
        0: "FX00_Sporadic_line_artefacts",
        1: "FX02_Group_of_line_artefact",
        2: "FX03_partly_brighter",
        3: "FX04_group_of_defect_line",
        4: "FX05_Defect_line",
        5: "FX07_Stripes",
        6: "Good_image",
    }
    return label_to_class[label]


def createPatch(image, nPatches):
    assert (
        image.shape[1] == 2866 and image.shape[2] == 2350
    ), f"expected image shape to be 2866*2350, instead got {image.shape}"
    assert math.sqrt(nPatches) == int(
        math.sqrt(nPatches)
    ), f"{nPatches} is not a perfect square"
    _, w, h = image.shape
    reshaped_image = image[:, 0:-1, 0:-1]
    prc = int(math.sqrt(nPatches))
    pw, ph = w // prc, h // prc
    imgPatches = [
        reshaped_image[
            :,
            (i % prc) * pw : ((i % prc) + 1) * pw,
            (i // prc) * ph : ((i // prc) + 1) * ph,
        ]
        for i in range(nPatches)
    ]
    return imgPatches


def import_and_predict(image_data):
    image_data = image_data.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint16), cv2.IMREAD_UNCHANGED)
    image_float32 = image.astype(np.float32)
    image_tensor = image_float32
    image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_GRAY2RGB)

    image_tensor = (image_tensor - image_tensor.min()) / (
        image_tensor.max() - image_tensor.min()
    )
    st.image(image_tensor, width=300)

    image_tensor = torch.from_numpy(image_tensor.astype(np.float32))

    image_tensor = image_tensor.permute(2, 0, 1)

    patch = createPatch(image_tensor, 9)
    prediction = []
    confidence = []
    for p in patch:
        output = model_ft(p.unsqueeze(0))

        probs = torch.nn.functional.softmax(output, dim=1)
        conf, preds = torch.max(probs, 1)
        # _, preds = torch.max(output, 1)
        print("output", preds)
        print("conf", conf)
        prediction.append(preds.item())
        confidence.append(conf.item())
    # take average of confidences
    print(prediction)
    final_prediction = classify_batch(prediction)
    # find mean of confidence where element in prediction matches  final_prediction
    confidence = [
        confidence[i]
        for i in range(len(prediction))
        if prediction[i] == final_prediction
    ]
    print(confidence)
    final_confidence = np.mean(confidence)
    return label_to_class(final_prediction), round(final_confidence, 5) * 100
