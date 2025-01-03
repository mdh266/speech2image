import replicate
import streamlit as st
import os
from google.cloud import speech


def generate_image(
    width: int, height: int, prompt: str
) -> replicate.helpers.FileOutput:
    output = replicate.run(
        "nvidia/sana:c6b5d2b7459910fec94432e9e1203c3cdce92d6db20f714f1355747990b52fa6",
        input={"width": width, "height": height, "prompt": prompt},
    )

    return output


@st.cache_data
def speech_to_text(
    google_api: str, audio_prompt: st.runtime.uploaded_file_manager.UploadedFile
) -> str:
    stt = speech.SpeechClient(client_options={"api_key": google_api})

    stt_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, language_code="en-US"
    )

    audio = speech.RecognitionAudio(content=audio_prompt.getvalue())

    response = stt.recognize(config=stt_config, audio=audio)

    text_prompt = response.results.pop().alternatives[0].transcript

    return text_prompt


def sidebar_form(google_api: str):
    with st.sidebar:
        with st.form("Form "):
            st.markdown("### Submit An Idea For An Image")

            width = st.slider(
                label="Image Width", min_value=100, max_value=2000, value=512
            )

            height = st.slider(
                label="Image Height", min_value=100, max_value=2000, value=512
            )

            audio_prompt = st.audio_input("Say your image idea!")
            submitted = st.form_submit_button("Submit")
            return submitted, width, height, audio_prompt


def main(debug: bool = False):
    if debug:
        from dotenv import load_dotenv
        load_dotenv()

    google_api = os.getenv("GOOGLE_API_KEY")

    st.markdown("# Speech To Image App")
    st.markdown(
        "Choose an image size and say the description of the image you want to create."
    )
    st.divider()

    submitted, width, height, audio_prompt = sidebar_form(google_api=google_api)

    if submitted:
        text_prompt = speech_to_text(google_api=google_api, audio_prompt=audio_prompt)

    if submitted and text_prompt:
        with st.expander("Your Idea Was..."):
            st.markdown(text_prompt)

        with st.spinner("Generating Your Image..."):
            image = generate_image(width=width, height=height, prompt=text_prompt)

        with st.container():
            st.image(image.read(), caption="Your Image", width=width)


if __name__ == "__main__":
    main(debug=False)
