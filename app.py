import streamlit as st
from PIL import Image
import pandas as pd
from model_helper import predict

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Car Damage Detection",
    page_icon="🚗",
    layout="wide"
)

# ------------------ CSS ------------------
st.markdown("""
<style>

/* Fade Animation */
@keyframes fadeIn {
    0% {opacity: 0; transform: translateY(15px);}
    100% {opacity: 1; transform: translateY(0);}
}

/* Glow */
@keyframes glowPulse {
    0% {box-shadow: 0 0 5px #00FFCC;}
    50% {box-shadow: 0 0 20px #00FFCC;}
    100% {box-shadow: 0 0 5px #00FFCC;}
}

/* Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: bold;
    color: #00FFCC;
    text-align: center;
    animation: fadeIn 1s ease-in-out;
}

/* Card */
.card {
    background-color: #1C1F26;
    padding: 20px;
    border-radius: 15px;
    animation: fadeIn 1s ease-in-out;
    transition: transform 0.3s ease;
}
.card:hover {
    transform: scale(1.02);
}

/* Prediction */
.prediction-box {
    background: linear-gradient(45deg, #00FFCC, #00ccaa);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    color: black;
    font-size: 26px;
    font-weight: bold;
    animation: fadeIn 1s ease-in-out, glowPulse 2s infinite;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="title">🚗 AI Car Damage Detection Dashboard</div>', unsafe_allow_html=True)
st.write("")

# ------------------ LAYOUT ------------------
col1, col2 = st.columns([1, 1])

# ------------------ LEFT SIDE ------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader("Choose car image", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width="stretch")

        image_path = "temp_file.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ RIGHT SIDE ------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction Result")

    if uploaded_file:

        with st.spinner("🔄 AI analyzing damage..."):
            prediction, confidence, all_probs = predict(image_path)

        # Prediction
        st.markdown(f"""
        <div class="prediction-box">
        🚨 {prediction}
        </div>
        """, unsafe_allow_html=True)

        # Confidence
        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence:.2f}**")

        # Status
        if confidence > 0.8:
            st.success("High Confidence ✅")
        elif confidence > 0.6:
            st.warning("Moderate Confidence ⚠️")
        else:
            st.error("Low Confidence ❌")

    else:
        st.info("Upload an image to see results")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ ANALYTICS SECTION ------------------
if uploaded_file:
    st.write("")
    st.markdown("### 📊 Model Insights")

    class_names = [
        'Front Breakage',
        'Front Crushed',
        'Front Normal',
        'Rear Breakage',
        'Rear Crushed',
        'Rear Normal'
    ]

    df = pd.DataFrame({
        "Class": class_names,
        "Probability": all_probs
    })

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📈 Probability Chart")
        st.bar_chart(df.set_index("Class"))

    with col4:
        st.subheader("📋 Detailed Values")
        st.dataframe(df, use_container_width=True)

# ------------------ FOOTER ------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Built with ❤️ | Deep Learning | ResNet50 | Streamlit Dashboard
</p>
""", unsafe_allow_html=True)