import streamlit as st
import subprocess
import sys
import tempfile
import os
import re

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Color-Invariant Saree Recognition",
    page_icon="🥻",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🥻 Color-Invariant Saree Recognition")
st.write("Upload a saree image to identify its type.")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "📷 Choose a saree image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    st.image(
        uploaded_file,
        caption="Uploaded Saree Image",
        use_container_width=True
    )

    # Prediction button
    if st.button("🔍 Predict Saree Type", use_container_width=True):

        # Temporary image file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            temp_file.write(uploaded_file.getbuffer())
            image_path = temp_file.name

        try:
            # Run prediction script
            result = subprocess.run(
                [
                    sys.executable,
                    "src/predict_resnet.py",
                    image_path
                ],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:

                output = result.stdout

                st.success("✅ Prediction Completed!")

                # -----------------------------
                # Extract probabilities
                # -----------------------------
                probabilities = {}

                for line in output.splitlines():

                    match = re.search(
                        r"(\w+)\s*:\s*([\d.]+)%",
                        line
                    )

                    if match:
                        saree_name = match.group(1)
                        confidence = float(match.group(2))
                        probabilities[saree_name] = confidence

                # -----------------------------
                # Extract prediction
                # -----------------------------
                prediction_match = re.search(
                    r"Prediction\s*:\s*(\w+)",
                    output,
                    re.IGNORECASE
                )

                if prediction_match:

                    prediction = prediction_match.group(1)

                    prediction = prediction.capitalize()

                    confidence = probabilities.get(
                        prediction.lower(),
                        0
                    )

                    # -----------------------------
                    # Main Result
                    # -----------------------------
                    st.subheader("🎯 Prediction Result")

                    st.success(
                        f"🥻 Predicted Saree: **{prediction}**"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.1f}%"
                    )

                    # -----------------------------
                    # Confidence Scores
                    # -----------------------------
                    st.subheader("📊 Confidence Scores")

                    for name, score in probabilities.items():

                        st.write(
                            f"**{name.capitalize()} — {score:.1f}%**"
                        )

                        st.progress(
                            min(int(score), 100)
                        )

                else:

                    # Fallback
                    st.subheader("Prediction Result")
                    st.code(output)

            else:

                st.error("❌ Prediction failed.")

                st.code(result.stderr)

        finally:

            # Delete temporary file
            if os.path.exists(image_path):
                os.remove(image_path)