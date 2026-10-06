import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌱",
    layout="wide"
)

# ==============================
# CUSTOM STYLE
# ==============================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================

st.markdown(
    '<div class="main-title">🌱 AgriVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Plant Disease Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a plant leaf image and our MobileNetV2 model "
    "will analyze it."
)

st.divider()

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "model/plant_disease_mobilenet.keras"
    )


model = load_model()

# ==============================
# CLASS NAMES
# ==============================

class_names = [
    "Pepper Bacterial Spot",
    "Pepper Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus"
]

# ==============================
# DISEASE INFORMATION
# ==============================

disease_info = {

    "Pepper Bacterial Spot": {
        "description": "A bacterial disease that causes dark spots on pepper leaves and fruits.",
        "treatment": "Remove infected plant material and avoid overhead watering.",
        "prevention": "Use healthy seeds and maintain good air circulation."
    },

    "Pepper Healthy": {
        "description": "The pepper plant appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Continue proper watering, nutrition and monitoring."
    },

    "Potato Early Blight": {
        "description": "A fungal disease that causes dark circular spots on potato leaves.",
        "treatment": "Remove severely infected leaves and use appropriate fungicide if necessary.",
        "prevention": "Avoid overhead watering and maintain good field sanitation."
    },

    "Potato Healthy": {
        "description": "The potato plant appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Continue regular monitoring and proper crop care."
    },

    "Potato Late Blight": {
        "description": "A serious disease that causes dark lesions on potato leaves.",
        "treatment": "Remove infected material and use suitable fungicide according to local guidance.",
        "prevention": "Improve air circulation and avoid prolonged leaf wetness."
    },

    "Tomato Bacterial Spot": {
        "description": "A bacterial disease causing small dark spots on tomato leaves and fruit.",
        "treatment": "Remove infected leaves and avoid working with wet plants.",
        "prevention": "Use disease-free seeds and maintain proper spacing."
    },

    "Tomato Early Blight": {
        "description": "A fungal disease that produces dark spots and yellowing on tomato leaves.",
        "treatment": "Remove infected leaves and consider an appropriate fungicide.",
        "prevention": "Rotate crops and avoid overhead irrigation."
    },

    "Tomato Healthy": {
        "description": "The tomato plant appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Continue regular watering, nutrition and monitoring."
    },

    "Tomato Late Blight": {
        "description": "A destructive disease causing dark lesions on tomato leaves.",
        "treatment": "Remove infected material and seek appropriate fungicide guidance.",
        "prevention": "Keep foliage dry and provide good air circulation."
    },

    "Tomato Leaf Mold": {
        "description": "A fungal disease that commonly develops under humid conditions.",
        "treatment": "Remove affected leaves and improve ventilation.",
        "prevention": "Reduce humidity and avoid wetting leaves."
    },

    "Tomato Septoria Leaf Spot": {
        "description": "A fungal disease producing many small spots on tomato leaves.",
        "treatment": "Remove infected leaves and improve sanitation.",
        "prevention": "Avoid overhead watering and remove plant debris."
    },

    "Tomato Spider Mites": {
        "description": "Spider mites can cause speckling, yellowing and leaf damage.",
        "treatment": "Use suitable mite-control methods and remove heavily affected leaves.",
        "prevention": "Monitor plants regularly and avoid excessive dryness."
    },

    "Tomato Target Spot": {
        "description": "A fungal disease producing circular target-like lesions.",
        "treatment": "Remove affected leaves and use appropriate disease-control measures.",
        "prevention": "Maintain good spacing and avoid prolonged leaf wetness."
    },

    "Tomato Mosaic Virus": {
        "description": "A viral disease that can cause mosaic patterns and abnormal leaf growth.",
        "treatment": "Remove infected plants because viral diseases have no simple cure.",
        "prevention": "Use clean tools and disease-free planting material."
    },

    "Tomato Yellow Leaf Curl Virus": {
        "description": "A viral disease that causes yellowing, curling and stunted growth.",
        "treatment": "Remove severely infected plants and control insect vectors.",
        "prevention": "Monitor and control whiteflies and use healthy planting material."
    }
}

# ==============================
# IMAGE UPLOAD
# ==============================

st.subheader("📷 Upload a Plant Leaf")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# PREDICTION
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

    with col2:

        st.info("🔍 Ready to analyze this image.")

        if st.button(
            "🌱 Analyze Leaf",
            use_container_width=True
        ):

            image_resized = image.resize((96, 96))

            image_array = np.array(
                image_resized,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            with st.spinner("AI is analyzing the leaf..."):

                prediction = model.predict(
                    image_array,
                    verbose=0
                )

            index = np.argmax(prediction[0])

            disease = class_names[index]

            confidence = (
                prediction[0][index] * 100
            )

            st.success("Analysis completed!")

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.subheader("🌿 Prediction")

            st.write(
                "**Detected condition:**",
                disease
            )

            st.write(
                "**Confidence:**",
                f"{confidence:.2f}%"
            )

            st.progress(
                min(int(confidence), 100)
            )

            st.markdown("</div>", unsafe_allow_html=True)

            # ==============================
            # LOW CONFIDENCE WARNING
            # ==============================

            if confidence < 60:

                st.warning(
                    "⚠️ The model is not very confident "
                    "about this prediction. Try uploading "
                    "a clearer leaf image."
                )

            # ==============================
            # INFORMATION
            # ==============================

            if disease in disease_info:

                info = disease_info[disease]

                st.divider()

                st.subheader("📋 About the Condition")

                st.write(
                    info["description"]
                )

                col3, col4 = st.columns(2)

                with col3:

                    st.subheader("💊 Treatment")

                    st.write(
                        info["treatment"]
                    )

                with col4:

                    st.subheader("🛡️ Prevention")

                    st.write(
                        info["prevention"]
                    )

st.divider()

st.caption(
    "AgriVision AI • Powered by MobileNetV2 • "
    "Plant Disease Detection"
)