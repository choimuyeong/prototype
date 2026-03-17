import streamlit as st

from src.mnist_service import ensure_model_exists, get_model_info, redownload_model, reload_session
from src.ui import apply_global_styles

st.set_page_config(page_title="모델 관리", layout="wide")
apply_global_styles()

st.title("모델 관리")
st.caption("MNIST ONNX 모델 상태를 확인하고 다운로드/세션 캐시를 관리합니다.")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("모델 상태 새로고침", use_container_width=False):
        st.rerun()

with col2:
    if st.button("모델 재다운로드", type="primary", use_container_width=False):
        with st.spinner("모델 파일을 다시 다운로드하는 중..."):
            path = redownload_model()
            reload_session()
        st.success(f"재다운로드 완료: {path.name}")

with col3:
    if st.button("세션 캐시 재로드", use_container_width=False):
        with st.spinner("ONNX 세션 캐시를 갱신하는 중..."):
            reload_session()
        st.success("세션 캐시 갱신 완료")

try:
    ensure_model_exists()
    info = get_model_info()
except Exception as exc:
    st.error(f"모델 준비 중 오류가 발생했습니다: {exc}")
    st.stop()

st.subheader("현재 모델 정보")
left, right = st.columns(2)

with left:
    st.metric("파일 존재", "예" if info["exists"] else "아니오")
    st.metric("파일 크기", f"{info['size_kb']} KB")

with right:
    st.write(f"경로: `{info['path']}`")
    st.write(f"최종 수정 시각: `{info['last_modified']}`")
