import streamlit as st

if "user" in st.session_state and st.session_state.user:
    st.info(f"{st.session_state.user}님 안녕하세요!")

st.header("입력 → 출력 폼 예제")

text = st.text_input("텍스트 입력")
k = st.slider("반복 횟수", 1, 5, 2)

if st.button("실행"):
    result = (text + " ") * k
    st.success(result)