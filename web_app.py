import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re
import time
import tempfile
import os
from io import BytesIO

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="AI PDF Extractor", page_icon="📄", layout="wide")

def inject_custom_css():
    st.markdown("""
        <style>
        /* Main Background and Text */
        .stApp {
            background-color: #FFFFFF;
            color: #333333;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #F5FAF6; /* Very light green tint */
            border-right: 1px solid #E0E0E0;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #009CA6; /* ASI Blue-Green / Teal */
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #009CA6;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #007A82; /* Darker teal on hover */
            color: white;
            border: none;
        }
        
        /* Inputs */
        .stTextInput > div > div > input {
            border-radius: 4px;
            border: 1px solid #CCCCCC;
            background-color: #FFFFFF; /* Ensure white background */
            color: #333333; /* Ensure dark text */
        }
        
        /* Data Editor */
        [data-testid="stDataFrame"] {
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            background-color: #FFFFFF;
        }
        
        /* Info Box */
        .stAlert {
            background-color: #F0F8F5;
            color: #005C61;
            border: 1px solid #009CA6;
        }
        
        /* Custom Header Bar */
        .header-bar {
            background-color: #009CA6; /* ASI Blue-Green */
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            background-image: linear-gradient(135deg, #009CA6 0%, #007A82 100%);
        }
        .header-bar h1 {
            color: white !important;
            margin-bottom: 0.5rem;
        }
        .header-bar p {
            font-size: 1.1rem;
            opacity: 0.95;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

DEFAULT_MODEL_NAME = "gemini-flash-latest"

NC_COLUMNS = {
    "NC number": "List every NC number recorded. If none, return 'None'.",
    "Client name": "List the client name(s) associated with the NCs. Use 'Unknown' if not stated.",
    "Indicator": "Copy the referenced Indicator in the standard for each NC.",
    "Grade": (
        "State the grade (Major, Minor, Critical, Non-critical, Opportunities for Improvement etc.) for each NC. "
        "If Opportunities for Improvement is mentioned, include it explicitly."
    ),
    "Status": (
        "Open or Closed. Open if there is no closed date, closed if there is one."
    ),
    "Issue Date": "Provide the date each NC was issued. Use the format (DD-MM-YYYY).",
    "Closed date": "Provide the date each NC was closed. Use the format (DD-MM-YYYY).",
    "Scope Definition": "Summarize the scope definition or description for each NC."
}

DEFAULT_NC_SCHEMA = [
    {"Column Name": column, "Question": description}
    for column, description in NC_COLUMNS.items()
]

def ensure_nc_schema_states():
    """Initialize per-section non-conformity schema tables."""
    for section in NC_SECTIONS:
        state_key = f"nc_schema_df_{section['key']}"
        if state_key not in st.session_state:
            st.session_state[state_key] = pd.DataFrame(DEFAULT_NC_SCHEMA)

NC_SECTIONS = [
    {
        "key": "audit_nc",
        "title": "4.3.1 Non-Conformities Identified during this Audit",
        "instruction": (
            "Focus strictly on Section 4.3.1 'Non-Conformities Identified during this Audit'. "
            "If multiple NCs exist, list them all in order."
        ),
        "file_name": "non_conformities_current_audit.csv"
    },
    {
        "key": "previous_nc",
        "title": "4.3.2 Non-Conformity Identified during the last ASA",
        "instruction": (
            "Focus strictly on Section 4.3.2 'Non-Conformity Identified during the last ASA'. "
            "If multiple NCs exist, list them all in order."
        ),
        "file_name": "non_conformities_last_asa.csv"
    }
]

def get_configured_api_key():
    """Load the API key from Streamlit secrets or environment variables."""
    secret_key = None
    try:
        secret_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        secret_key = None
    return secret_key or os.environ.get("GEMINI_API_KEY")

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




def _clean_json_payload(raw_text):
    """Strip markdown code fences and whitespace before JSON parsing."""
    payload = (raw_text or "").strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?", "", payload, flags=re.IGNORECASE).strip()
        payload = re.sub(r"```$", "", payload).strip()
    return payload


def generate_gemini_schema(schema_dict):
    """Converts user schema to Gemini response schema."""
    properties = {}
    required = []
    for col_name, question in schema_dict.items():
        properties[col_name] = {
            "type": "STRING",
            "description": question
        }
        required.append(col_name)
    
    return {
        "type": "OBJECT",
        "properties": properties,
        "required": required
    }


def analyze_document_with_gemini(filename, file_path, schema_dict, api_key, model_name, extra_instruction=None):
    """Uploads PDF to Gemini and extracts information."""
    try:
        genai.configure(api_key=api_key)
        
        # Upload file to Gemini
        uploaded_file = genai.upload_file(path=file_path, display_name=filename)
        
        # Poll for processing completion
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
            
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError("Gemini file processing failed.")
            
        # Generate schema for structured output
        schema = generate_gemini_schema(schema_dict)
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        )

        prompt = "Extract the information from the provided document according to the JSON schema. Ensure all fields are populated."
        if extra_instruction:
            prompt += f" {extra_instruction}"

        response = model.generate_content([uploaded_file, prompt])
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

st.markdown("""
    <div class="header-bar">
        <h1>📄 AI PDF Data Extractor</h1>
        <p>Upload PDFs, define your questions, and let AI convert them into a structured Excel/CSV table.</p>
    </div>
""", unsafe_allow_html=True)

st.info("💡 The questions you define below determine the columns in your output file.")

API_KEY = get_configured_api_key()
MODEL_NAME = DEFAULT_MODEL_NAME

if not API_KEY:
    st.warning(
        "Set `GEMINI_API_KEY` in Streamlit secrets (`.streamlit/secrets.toml`) or as an environment variable "
        "before running the extractor."
    )

# --- Main Area: Schema Definition ---
st.subheader("1. Define your Data Columns")
st.markdown("Edit the table below to choose what information to extract.")

schema_upload = st.file_uploader(
    "Optional: Upload a schema CSV (`Column Name`, `Question`)",
    type=["csv"],
    key="schema_file_uploader",
    help="Uploading a CSV will replace the table below."
)

if schema_upload is not None:
    try:
        uploaded_schema = pd.read_csv(schema_upload)
        required_cols = {"Column Name", "Question"}
        if not required_cols.issubset(uploaded_schema.columns):
            st.error("CSV must include `Column Name` and `Question` columns.")
        elif uploaded_schema.empty:
            st.error("Uploaded schema CSV is empty.")
        else:
            st.session_state.schema_df = uploaded_schema[["Column Name", "Question"]]
            st.success("Schema table updated from CSV upload.")
    except Exception as csv_err:
        st.error(f"Failed to read schema CSV: {csv_err}")

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
scan_nc = st.checkbox("Also scan for non-conformities?", value=False,
                      help="Runs additional prompts to scan and export data for non-conformities.")
if scan_nc:
    ensure_nc_schema_states()
    st.subheader("Configure Non-Conformity Questions")
    for section in NC_SECTIONS:
        state_key = f"nc_schema_df_{section['key']}"
        st.markdown(f"**{section['title']}**")
        nc_schema_upload = st.file_uploader(
            f"Upload schema CSV for {section['title']} (`Column Name`, `Question`)",
            type=["csv"],
            key=f"nc_schema_file_uploader_{section['key']}",
            help="Uploading replaces the table below for this section."
        )
        if nc_schema_upload is not None:
            try:
                nc_schema_df = pd.read_csv(nc_schema_upload)
                required_cols = {"Column Name", "Question"}
                if not required_cols.issubset(nc_schema_df.columns):
                    st.error("NC schema CSV must include `Column Name` and `Question` columns.")
                elif nc_schema_df.empty:
                    st.error("Uploaded NC schema CSV is empty.")
                else:
                    st.session_state[state_key] = nc_schema_df[["Column Name", "Question"]]
                    st.success(f"Updated {section['title']} questions from CSV.")
            except Exception as nc_upload_err:
                st.error(f"Failed to read NC schema CSV for {section['title']}: {nc_upload_err}")

        current_nc_df = st.session_state[state_key]
        edited_nc_df = st.data_editor(
            current_nc_df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"nc_schema_editor_{section['key']}"
        )
        st.session_state[state_key] = edited_nc_df
        st.markdown("---")

# --- Main Area: Execution ---
if st.button("🚀 Start Extraction", type="primary"):
    if not API_KEY:
        st.error("Missing API key. Configure `GEMINI_API_KEY` via secrets or environment variable.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    elif edited_schema.empty:
        st.error("Please define at least one question to extract.")
    elif scan_nc and any(
        st.session_state[f"nc_schema_df_{section['key']}"].empty for section in NC_SECTIONS
    ):
        st.error("Each non-conformity schema must contain at least one row. Update the tables above before continuing.")
    else:
        schema_dict = pd.Series(
            edited_schema.Question.values, index=edited_schema['Column Name']
        ).to_dict()

        results_section = st.container()
        with results_section:
            st.subheader("3. Extracted Data")
            main_table_placeholder = st.empty()

        nc_table_placeholders = {}
        if scan_nc:
            nc_section_container = st.container()
            with nc_section_container:
                st.subheader("4. Non-Conformities")
                for section in NC_SECTIONS:
                    st.markdown(f"**{section['title']}**")
                    nc_table_placeholders[section["key"]] = st.empty()

        def process_documents():
            results = []
            nc_results = {section["key"]: [] for section in NC_SECTIONS} if scan_nc else {}
            nc_schema_dicts = {}
            if scan_nc:
                for section in NC_SECTIONS:
                    df = st.session_state[f"nc_schema_df_{section['key']}"]
                    nc_schema_dicts[section["key"]] = pd.Series(
                        df.Question.values,
                        index=df['Column Name']
                    ).to_dict()
            progress_bar = st.progress(0)
            status_text = st.empty()

            main_table_df = None
            nc_table_dfs = {section["key"]: None for section in NC_SECTIONS} if scan_nc else {}

            def update_main_table_display():
                nonlocal main_table_df
                if not results:
                    main_table_placeholder.info("Waiting for documents...")
                    main_table_df = None
                    return
                df = pd.DataFrame(results)
                cols = ['filename'] + [k for k in schema_dict.keys() if k != 'filename']
                for col in cols:
                    if col not in df.columns:
                        df[col] = "N/A"
                df = df[cols]
                main_table_placeholder.dataframe(df, use_container_width=True)
                main_table_df = df

            def update_nc_table_display(section_key):
                if not scan_nc:
                    return
                rows = nc_results.get(section_key, [])
                placeholder = nc_table_placeholders[section_key]
                if not rows:
                    placeholder.info("Waiting for documents...")
                    nc_table_dfs[section_key] = None
                    return
                df = pd.DataFrame(rows)
                cols = ['filename'] + list(nc_schema_dicts[section_key].keys())
                for col in cols:
                    if col not in df.columns:
                        df[col] = "N/A"
                df = df[cols]
                placeholder.dataframe(df, use_container_width=True)
                nc_table_dfs[section_key] = df

            for i, pdf_file in enumerate(uploaded_files):
                status_text.text(f"Processing {pdf_file.name}...")
                
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    extracted_data = analyze_document_with_gemini(
                        pdf_file.name, tmp_file_path, schema_dict, API_KEY, MODEL_NAME
                    )
                    results.append(extracted_data)
                    update_main_table_display()

                    if scan_nc:
                        for section in NC_SECTIONS:
                            section_schema = {
                                col: f"{desc} (Context: {section['title']})."
                                for col, desc in nc_schema_dicts[section["key"]].items()
                            }
                            try:
                                section_data = analyze_document_with_gemini(
                                    pdf_file.name,
                                    tmp_file_path,
                                    section_schema,
                                    API_KEY,
                                    MODEL_NAME,
                                    extra_instruction=section["instruction"]
                                )
                                nc_results[section["key"]].append(section_data)
                                update_nc_table_display(section["key"])
                            except Exception as nc_err:
                                st.error(f"Error processing non-conformities for {pdf_file.name} ({section['title']}): {nc_err}")
                                error_row = {
                                    "NC number": "Extraction Error",
                                    "Client name": "Extraction Error",
                                    "filename": pdf_file.name
                                }
                                nc_results[section["key"]].append(error_row)
                                update_nc_table_display(section["key"])
                except Exception as e:
                    st.error(f"Error processing {pdf_file.name}: {e}")
                    error_row = {key: "Extraction Error" for key in schema_dict.keys()}
                    error_row['filename'] = pdf_file.name
                    results.append(error_row)
                    update_main_table_display()
                finally:
                    # Clean up the temporary file
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                time.sleep(0.5)

            status_text.text("Done!")

            # --- Export / Save ---
            if main_table_df is not None:
                csv = main_table_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="extracted_data.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No data was extracted.")

            if scan_nc:
                for section in NC_SECTIONS:
                    section_df = nc_table_dfs.get(section["key"])
                    if section_df is not None and not section_df.empty:
                        csv_data = section_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"📥 Download CSV - {section['title']}",
                            data=csv_data,
                            file_name=section["file_name"],
                            mime="text/csv",
                        )

        process_documents()
