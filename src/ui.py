import streamlit as st


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a * {
            font-size: 20px !important;
            font-weight: 800 !important;
            line-height: 1.35 !important;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"],
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] * {
            font-size: 24px !important;
            font-weight: 900 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    