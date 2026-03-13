import streamlit as st

from src.ui import apply_global_styles

st.set_page_config(page_title="MNIST Inference Service", layout="wide")
apply_global_styles()

st.title("MNIST기반 손글씨 숫자 인식 서비스")
st.caption("사이드바에서 페이지를 선택해 기능을 사용하세요.")
st.caption("웹 배포 실습을 위한 prototype입니다.")

st.subheader("현재 구성")
st.write("- 입력 캔버스 + 전처리 이미지")
st.write("- 모델 추론 결과")
st.write("- 이미지 저장소")
st.write("- 모델 관리")
