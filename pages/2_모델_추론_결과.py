import altair as alt
import streamlit as st

from src.mnist_service import predict
from src.ui import apply_global_styles

st.set_page_config(page_title="모델 추론 결과", layout="wide")
apply_global_styles()

st.title("모델 추론 결과")
st.caption("입력/전처리 페이지에서 만든 이미지를 기준으로 예측합니다.")

img = st.session_state.get("preprocessed_28x28")
has_drawing = st.session_state.get("has_drawing", False)

if img is None or not has_drawing:
    st.warning("먼저 '입력 + 전처리' 페이지에서 숫자를 그려주세요.")
    st.stop()

pred, probs = predict(img)

# 다운로드용 데이터 보관
st.session_state["latest_infer_image"] = img
st.session_state["latest_infer_pred"] = int(pred)
st.session_state["latest_infer_probs"] = probs.tolist()

left, right = st.columns([1, 2])

with left:
    st.subheader("전처리 입력")
    st.image(img, clamp=True, width=220)
    st.metric("예측 숫자", pred)
    st.metric("신뢰도", f"{probs[pred] * 100:.2f}%")
    st.info("저장은 '이미지 저장소' 페이지에서 할 수 있습니다.")

with right:
    st.subheader("레이블별 확률")
    chart_data = [{"label": str(i), "prob": float(p)} for i, p in enumerate(probs)]
    chart = (
        alt.Chart(alt.Data(values=chart_data))
        .mark_bar()
        .encode(
            x=alt.X("label:N", title="레이블", axis=alt.Axis(labelAngle=0, labelFontSize=16, titleFontSize=18)),
            y=alt.Y("prob:Q", title="확률", axis=alt.Axis(titleAngle=0, labelFontSize=14, titleFontSize=16), scale=alt.Scale(domain=[0, 1])),
            tooltip=["label:N", alt.Tooltip("prob:Q", format=".4f")],
        )
        .properties(width=800, height=500)
    )
    st.altair_chart(chart, use_container_width=False)
