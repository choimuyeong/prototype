import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.mnist_service import preprocess_canvas
from src.ui import apply_global_styles

st.set_page_config(page_title="입력 + 전처리", layout="wide")
apply_global_styles()

st.title("입력 캔버스 + 전처리 이미지")
st.caption("숫자를 그리면 모델 입력 형태(28x28)에 맞춘 전처리 결과를 함께 보여줍니다.")

left, right = st.columns(2)

with left:
    st.subheader("입력 캔버스")
    stroke_width = st.session_state.get("stroke_width", 10)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="input_canvas",
    )
    slider_col, _ = st.columns([2, 3])
    with slider_col:
        st.slider("선 굵기", min_value=1, max_value=20, value=stroke_width, step=1, key="stroke_width")

with right:
    st.subheader("전처리 이미지")
    if canvas_result.image_data is None:
        st.info("왼쪽 캔버스에 숫자를 그려주세요.")
        st.session_state["preprocessed_28x28"] = None
    else:
        preprocessed = preprocess_canvas(canvas_result.image_data)
        st.image(preprocessed, clamp=True, width=280)
        st.caption("28x28 grayscale")
        st.session_state["preprocessed_28x28"] = preprocessed
        st.session_state["has_drawing"] = bool(preprocessed.max() > 0.0)
