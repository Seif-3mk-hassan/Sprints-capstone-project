import streamlit as st
import requests
import json

# Backend URL
BACKEND_URL = "http://127.0.0.1:8001/generate"

st.set_page_config(page_title="SwiftDesk AI Assistant", layout="wide")

st.title("SwiftDesk IT Support Assistant")

# =========================
# User Input Section
# =========================
st.subheader("Enter Customer Issue")

customer_issue = st.text_area(
    "Describe the issue:",
    placeholder="Example: My laptop cannot connect to WiFi..."
)

# =========================
# Controls
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    prompt_style = st.selectbox(
        "Prompt Style",
        ["zero-shot", "few-shot", "reasoned"]
    )

with col2:
    use_rag = st.checkbox("Enable RAG", value=True)

with col3:
    k_examples = st.slider("Top-K Retrieved", 1, 10, 3)

# =========================
# Generate Button
# =========================
if st.button("Generate Reply"):
    if not customer_issue.strip():
        st.warning("Please enter a customer issue.")
    else:
        with st.spinner("Generating response..."):
            payload = {
            "issue": customer_issue,
            "use_rag": use_rag,
            "prompt_style": prompt_style,
            "k": k_examples,
            "return_retrieval": use_rag   
        }

            try:
                response = requests.post(BACKEND_URL, json=payload)
                if response.status_code != 200:
                    st.error(response.text)
                else:
                    data = response.json()

                # =========================
                # Output
                # =========================
                st.subheader("Generated Reply")
                st.write(data.get("reply", "No reply generated."))

                # =========================
                # Show Retrieved Examples
                # =========================
                if use_rag and "sources" in data:
                    st.subheader("Retrieved Support Examples")

                    for i, src in enumerate(data["sources"]):
                        with st.expander(f"Example {i+1}"):
                            st.write("**Issue:**", src.get("issue"))
                            st.write("**Reply:**", src.get("reply"))

            except Exception as e:
                st.error(f"Error: {e}")