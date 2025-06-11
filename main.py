import streamlit as st
import requests

# Set up the page
st.set_page_config(page_title="🔐 QSAFAgent Runtime Monitor", layout="centered")

st.markdown("## 🔐 QSAFAgent Runtime Monitor")
st.markdown("Monitor AI threats across 9 security domains in real time.")

# API Endpoint (change if deployed elsewhere)
API_URL = "http://localhost:5000/status"

# Form inputs
with st.form("qsaf_input_form"):
    st.markdown("### 📝 Prompt")
    prompt = st.text_input("", value="ignore previous instructions")

    st.markdown("### 📌 Context")
    context = st.text_input("", value="financial")

    st.markdown("### 🔌 Plugin")
    plugin_name = st.text_input("", value="web_search")

    st.markdown("### 🎯 User Intent")
    user_intent = st.text_input("", value="fetch_data")

    st.markdown("### 🧠 LLM Response")
    response = st.text_area("", value="Sample response with unknown data")

    st.markdown("### 📄 Retrieved Docs")
    docs_input = st.text_input("", value="doc1,doc2")

    submit_button = st.form_submit_button("🛡️ Run QSAF Analysis")

# On Submit
if submit_button:
    try:
        payload = {
            "prompt": prompt,
            "context": context,
            "plugin_name": plugin_name,
            "user_intent": user_intent,
            "response": response,
            "docs": [d.strip() for d in docs_input.split(",")]
        }
        response = requests.post(API_URL, json=payload)
        result = response.json()

        st.success("✅ Analysis Completed")

        # Expandable results by domain
        with st.expander("🛡️ Prompt Injection Detection"):
            st.json(result.get("prompt_result", {}))

        with st.expander("🔐 Role & Context Manipulation"):
            st.json(result.get("role_manipulation", {}))

        with st.expander("🔌 Plugin Governance"):
            st.json({"plugin_allowed": result.get("plugin_allowed", None)})

        with st.expander("⚠️ Output Risk Analysis"):
            st.json(result.get("output_risk", {}))

        with st.expander("📊 Behavioral Anomaly Detection"):
            st.json(result.get("anomaly_result", {}))

        with st.expander("🔏 Payload Integrity & Signing"):
            st.code(result.get("signed_payload", ""), language="json")

        with st.expander("📚 RAG Source Attribution"):
            st.json(result.get("rag_monitoring", {}))

        with st.expander("🗃️ Data Governance"):
            st.json(result.get("data_governance", {}))

        with st.expander("🌐 Cross-Environment Defense"):
            st.json(result.get("cross_defense", {}))

    except Exception as e:
        st.error(f"❌ Error while contacting backend: {e}")
