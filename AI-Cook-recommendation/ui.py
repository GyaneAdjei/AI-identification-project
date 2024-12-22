import cv2
import numpy as np
import streamlit as st

from inference import inference, initialization
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from pyzbar.pyzbar import decode as qr_decode

metadata = MetadataCatalog.get("_")
metadata.thing_classes = ['None', 'qr_code']

def decoder(image):
    gray_img = cv2.cvtColor(image,0)
    qr = qr_decode(gray_img)[0]

    qrCodeData = qr.data.decode("utf-8")
    return qrCodeData

@st.cache(persist=True)
def draw_img(img, metadata, outputs):
    v = Visualizer(img[:, :, ::-1],metadata=metadata, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()

def main():

    predictor = initialization()

    # Streamlit initialization
    html_temp = """
        <div style="background-color:black;padding:5px">
        <h2 style="color:white;text-align:center;">QR code scanner and reader</h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Create a FileUploader so that the user can upload an image to the UI
    uploaded_file = st.file_uploader(label="Upload an image",
                                 type=["png", "jpeg", "jpg"])

    # Display the predict button just when an image is being uploaded
    if not uploaded_file:
        st.warning("Please upload an image before proceeding!")
        st.stop()
    else:
        # image_as_bytes = uploaded_file.read()
        image_as_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(image_as_bytes, 1)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        st.image(img_cv, width=500)
        pred_button = st.button("Predict")
    
    if pred_button:

        outputs = inference(predictor, img)
        out_img = draw_img(img, metadata,outputs)
        st.image(out_img, width=700)   

        instances = outputs["instances"]
        for i in range(len(instances)):  # Loop through each detected instance
            instance = instances[i]
            box = instance.pred_boxes.tensor[0].cpu().numpy()  # Get the bounding box coordinates
            x1, y1, x2, y2 = box
            cropped_object = img[int(y1):int(y2), int(x1):int(x2)]
            st.image(cropped_object[:, :, ::-1], width=500)
        
        st.text(decoder(cropped_object))

if __name__ == '__main__':
    main()