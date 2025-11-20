import streamlit as st
import pandas as pd
import pypdf
import google.generativeai as genai
import json
import re
import time
from io import BytesIO

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="AI PDF Extractor", page_icon="📄", layout="wide")

DEFAULT_MODEL_NAME = "gemini-flash-latest"

# Default schema to populate the editor
DEFAULT_SCHEMA = [
    {
        "Column Name": "CH ID",
        "Question": "Used to link together the CH sheet with the Audits sheet. Should at all times be kept unique and unchanged for a particular CH. (to be generated, not from the report)"
    },
    {"Column Name": "Report", "Question": "Name of the audit report"},
    {"Column Name": "Audit ID", "Question": "Unique code for the audit report from which the CH data is collected"},
    {"Column Name": "Certificate holder", "Question": "Full name extracted from the RSPO certificate"},
    {"Column Name": "Certified Mill Name", "Question": "Name of the mill"},
    {"Column Name": "Certified Mill's Location/Address", "Question": "Address of the mill"},
    {"Column Name": "Country", "Question": "A data standard should be followed"},
    {"Column Name": "Province", "Question": "Used to provide extra differentiation, as there are a lot of differences between provinces."},
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_text_from_pdf(uploaded_file):
    """Reads text from a Streamlit uploaded file object."""
    text_content = []
    try:
        reader = pypdf.PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_content.append(page_text)
        return "\n".join(text_content)
    except Exception as e:
        return f"Error reading PDF: {e}"


def _clean_json_payload(raw_text):
    """Strip markdown code fences and whitespace before JSON parsing."""
    payload = (raw_text or "").strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?", "", payload, flags=re.IGNORECASE).strip()
        payload = re.sub(r"```$", "", payload).strip()
    return payload


def analyze_document_with_gemini(filename, text, schema_dict, api_key, model_name):
    """Sends text to Gemini for extraction."""
    if not text.strip():
        return {key: "Error: Empty Document" for key in schema_dict.keys()}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )

        prompt = f"""
        Extract the following information from the document text.
        Return the output strictly as a JSON object.
        For missing info, use "N/A".
        
        FIELDS TO EXTRACT:
        {json.dumps(schema_dict, indent=2)}
        
        DOCUMENT TEXT:
        {text[:30000]} 
        """

        response = model.generate_content(prompt)
        response_payload = _clean_json_payload(getattr(response, "text", "") or "")
        data = json.loads(response_payload)
        data['filename'] = filename
        return data

    except Exception as e:
        # Return error dict to keep table alignment
        err_row = {key: "Extraction Error" for key in schema_dict.keys()}
        err_row['filename'] = filename
        st.error(f"Gemini extraction failed for {filename}: {e}")
        return err_row

# ==========================================
# APP UI
# ==========================================

st.title("📄 AI PDF Data Extractor")
st.markdown("Upload PDFs, define your questions, and let AI convert them into a structured Excel/CSV table.")

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password", help="Get one at aistudio.google.com")
    model_name = st.text_input(
        "Gemini Model",
        value=DEFAULT_MODEL_NAME,
        help="Use the full model name, e.g. models/gemini-1.5-flash"
    )
    
    st.info("💡 **Tip:** The questions you define in the main view determine the columns in your output file.")

# --- Main Area: Schema Definition ---
st.subheader("1. Define your Data Columns")
st.markdown("Edit the table below to choose what information to extract.")

if "schema_df" not in st.session_state:
    st.session_state.schema_df = pd.DataFrame(DEFAULT_SCHEMA)

edited_schema = st.data_editor(
    st.session_state.schema_df,
    num_rows="dynamic",
    use_container_width=True,
    key="schema_editor"
)

# --- Main Area: File Upload ---
st.subheader("2. Upload Documents")
uploaded_files = st.file_uploader("Drag and drop PDF files here", type=["pdf"], accept_multiple_files=True)

# --- Main Area: Execution ---
if st.button("🚀 Start Extraction", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not model_name:
        st.error("Please enter a Gemini model name in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    elif edited_schema.empty:
        st.error("Please define at least one question to extract.")
    else:
        schema_dict = pd.Series(
            edited_schema.Question.values, index=edited_schema['Column Name']
        ).to_dict()

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, pdf_file in enumerate(uploaded_files):
            status_text.text(f"Processing {pdf_file.name}...")
            
            raw_text = extract_text_from_pdf(pdf_file)

            if raw_text.startswith("Error reading PDF:"):
                st.error(f"{pdf_file.name}: {raw_text}")
                error_row = {key: "Extraction Error" for key in schema_dict.keys()}
                error_row['filename'] = pdf_file.name
                results.append(error_row)
                progress_bar.progress((i + 1) / len(uploaded_files))
                continue
            
            extracted_data = analyze_document_with_gemini(
                pdf_file.name, raw_text, schema_dict, api_key, model_name
            )
            results.append(extracted_data)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            time.sleep(0.5)

        status_text.text("Done!")
        
        # --- Results Display ---
        st.subheader("3. Extracted Data")
        
        if results:
            df = pd.DataFrame(results)
            
            cols = ['filename'] + [k for k in schema_dict.keys() if k != 'filename']
            for col in cols:
                if col not in df.columns:
                    df[col] = "N/A"
            df = df[cols]

            st.dataframe(df, use_container_width=True)

            # --- Export / Save ---
            csv = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="extracted_data.csv",
                mime="text/csv",
            )
        else:
            st.warning("No data was extracted.")
