import io
import json
from datetime import datetime

import streamlit as st
from PIL import Image

from src.ui import apply_global_styles

st.set_page_config(page_title="이미지 저장소", layout="wide")
apply_global_styles()

st.title("이미지 저장소")
st.caption("가장 최근 추론 결과를 로컬 파일로 저장할 수 있습니다.")

img = st.session_state.get("latest_infer_image")
pred = st.session_state.get("latest_infer_pred")
probs = st.session_state.get("latest_infer_probs")

if img is None or pred is None or probs is None:
    st.warning("먼저 '모델 추론 결과' 페이지에서 추론을 실행해주세요.")
    st.stop()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
probs = [float(p) for p in probs]

tab1, tab2, tab3 = st.tabs(["이미지 다운로드", "추론결과 다운로드", "확률표 다운로드"])

with tab1:
    st.image(img, clamp=True, width=220)
    png_buffer = io.BytesIO()
    Image.fromarray((img * 255).astype("uint8")).save(png_buffer, format="PNG")
    st.download_button(
        "PNG 다운로드",
        data=png_buffer.getvalue(),
        file_name=f"mnist_preprocessed_{ts}.png",
        mime="image/png",
        use_container_width=False,
    )

with tab2:
    payload = {
        "timestamp": ts,
        "predicted_label": int(pred),
        "confidence": round(float(probs[pred]), 4),
        "probabilities": [round(float(p), 4) for p in probs],
    }
    st.json(payload)
    st.download_button(
        "JSON 다운로드",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"mnist_result_{ts}.json",
        mime="application/json",
        use_container_width=False,
    )

with tab3:
    csv_text = "label,probability\n" + "\n".join([f"{i},{p}" for i, p in enumerate(probs)])
    st.dataframe(
        [{"label": i, "probability": p} for i, p in enumerate(probs)],
        width=250,
        hide_index=True,
    )
    st.download_button(
        "CSV 다운로드",
        data=csv_text.encode("utf-8"),
        file_name=f"mnist_probs_{ts}.csv",
        mime="text/csv",
        use_container_width=False,
    )
