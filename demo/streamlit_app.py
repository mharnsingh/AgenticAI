import streamlit as st
import requests
import os

# Use the API URL from env variable or default to localhost
api_url = os.getenv("AGENT_API_URL", "http://localhost:8000/query")

st.title("AI Agent Demo")

query = st.text_area("Enter your query:", "")

if st.button("Submit Query"):
    if not query.strip():
        st.error("Please enter a query.")
    else:
        try:
            response = requests.post(api_url, json={"query": query})
            if response.status_code == 200:
                result = response.json().get("result", {})
                st.success("Response:")
                st.json(result)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")