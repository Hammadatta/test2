import streamlit as st
import requests
import json

st.set_page_config(page_title="QSAFAgent Runtime Monitor", layout="wide")

st.markdown("""
# 🔐 QSAFAgent Runtime Monitor  
Monitor AI threats across 9 security domains in real time.
""")

# Backend API URL (Change this if deployed)
API_URL = "http://localhost:5000/status"

with st.form("qsaf_form"):
    prompt = st.text_input("📝 Prompt", value="ignore previous instructions")
    context = st.text_input("📌 Context", value="financial")
    plugin_name = st.text_input("🔌 Plugin", value="web_search")
    user_intent = st.text_input("🎯 User Intent", value="fetch_data")
    response = st.text_area("🧠 LLM Response", value="Sample response with unknown data")
    docs = st.text_area("📄 Retrieved Docs", value="doc1,doc2")

    submitted = st.form_submit_button("Run QSAF Analysis")

if submitted:
    try:
        payload = {
            "prompt": prompt,
            "context": context,
            "plugin_name": plugin_name,
            "user_intent": user_intent,
            "response": response,
            "docs": docs.split(",")
        }
        res = requests.post(API_URL, json=payload)
        result = res.json()

        st.success("✅ Analysis Completed")
        st.json(result)

        # Optional tabs
        tab1, tab2, tab3 = st.tabs(["Prompt Injection", "Output Risk", "Payload Signature"])
        with tab1:
            st.subheader("🛡️ Prompt Injection Detection")
            st.json(result.get("prompt_result", {}))
        with tab2:
            st.subheader("⚠️ Output Risk Evaluation")
            st.json(result.get("output_risk", {}))
        with tab3:
            st.subheader("🔏 Signed Payload Token")
            st.code(result.get("signed_payload", ""), language="json")

    except Exception as e:
        st.error(f"❌ Error: {e}")
