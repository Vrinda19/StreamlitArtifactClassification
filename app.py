import streamlit as st
import torch
from torchvision import models
import torch
from predict import import_and_predict

st.set_option("deprecation.showfileUploaderEncoding", False)


# @st.cache_data(allow_output_mutation=True)
def load_model(ckpt_path, num_classes):
    model_ft = models.resnet18()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    # model_ft=torch.load(ckpt_path)['model_state_dict']
    state = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model_ft.load_state_dict(state["model_state_dict"])
    model_ft.eval()
    return model_ft


model_ft = load_model(
    "your model path",
    7,
)

# st.title("Artifact Classification")
# st.header("Upload a TIF image for image classification")

# Use HTML/CSS styling to make the font size smaller
st.markdown(
    "<h1 style='font-size: 22px;'>Artifact Classification</h1>", unsafe_allow_html=True
)
st.markdown(
    "<h2 style='font-size: 16px;'>Upload a TIF image for image classification</h2>",
    unsafe_allow_html=True,
)

file = st.file_uploader("", type=["tif"])

if file is None:
    st.markdown(
        "<p style='font-size: 14px;'>Please upload an image file</p>",
        unsafe_allow_html=True,
    )
else:
    prediction, conf = import_and_predict(file)
    string = (
        "Image is classified as "
        + str(prediction)
        + " with confidence "
        + str(conf)
        + "%"
    )
    st.success(string)
