import base64

import streamlit as st
from workspaces.apis import request_inference


def create_inference_ui():
    st.title("KServe Inference Service")

    # Service configuration inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        inference_service_url = st.text_input("Inference Service URL")
    with col2:
        service_hostname = st.text_input("Service Hostname")
    with col3:
        model_name = st.text_input("Model Name")

    # Image upload
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Dynamic text input list
    if "text_inputs" not in st.session_state:
        st.session_state.text_inputs = [""]

    # Function to add new text input
    def add_text_input():
        st.session_state.text_inputs.append("")

    # Function to remove text input
    def remove_text_input(index):
        st.session_state.text_inputs.pop(index)

    # Text inputs with add/remove buttons
    st.subheader("Text Inputs")
    for i, text in enumerate(st.session_state.text_inputs):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.session_state.text_inputs[i] = st.text_input(f"Text {i+1}", value=text, key=f"text_input_{i}")
        with col2:
            if st.button("Remove", key=f"remove_{i}"):
                remove_text_input(i)
                st.rerun()

    st.button("Add Text Input", on_click=add_text_input)

    # Submit button
    if st.button("Run Inference"):
        if not uploaded_file:
            st.error("Please upload an image")
            return

        if not all([inference_service_url, service_hostname, model_name]):
            st.error("Please fill in all service configuration fields")
            return

        # Filter out empty text inputs
        text_list = [text for text in st.session_state.text_inputs if text.strip()]

        try:
            with st.spinner("Running inference..."):
                # Reset file pointer to beginning
                uploaded_file.seek(0)

                response = request_inference(
                    inference_service_url=inference_service_url,
                    service_hostname=service_hostname,
                    model_name=model_name,
                    image=uploaded_file.getvalue(),
                    text=text_list,
                )

                st.success("Inference completed!")
                print(f"response = {type(response)}")
                image = response
                image_bytes = base64.b64decode(image)

                st.image(image_bytes, caption="obejct detection image", use_column_width=True)
                # st.json(response)
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")


create_inference_ui()
