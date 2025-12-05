import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re
import time
import tempfile
import os
import base64
from io import BytesIO
import pdfplumber
from streamlit_pdf_viewer import pdf_viewer

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="AI PDF Extractor", page_icon="📄", layout="wide")

# Helper to load image as base64
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

def inject_custom_css():
    st.markdown("""
        <style>
        /* Main Background and Text */
        .stApp {
            background-color: #F5F5F5;
            color: #1D2C30;
            font-family: ui-sans-serif, system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
        }

        /* Top Header Bar */
        header[data-testid="stHeader"] {
            background-color: #204855 !important;
            z-index: 1000;
        }
        /* Top Header Buttons (Deploy, etc.) */
        header[data-testid="stHeader"] button {
            color: white !important;
        }
        header[data-testid="stHeader"] svg {
            fill: white !important;
            color: white !important;
        }

        /* Fixed Logo in Top Left */
        .fixed-logo {
            position: fixed;
            top: 0.5rem;
            left: 1rem;
            height: 2.5rem;
            z-index: 999999; /* Ensure it's above the header */
        }
        
        /* Sidebar - Hidden as requested */
        [data-testid="stSidebar"] {
            display: none;
        }
        section[data-testid="stSidebar"] {
            display: none;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #204855; /* ASI Dark Teal */
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #204855;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #325A67; /* ASI Accent */
            color: white;
            border: none;
        }
        
        /* Inputs */
        .stTextInput > div > div > input {
            border-radius: 4px;
            border: 1px solid #CCCCCC;
            background-color: #FFFFFF;
            color: #1D2C30;
        }
        
        /* Data Editor */
        [data-testid="stDataFrame"] {
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            background-color: #FFFFFF;
        }
        
        /* Info Box */
        .stAlert {
            background-color: #E8F1F2;
            color: #1D2C30;
            border: 1px solid #204855;
        }
        
        /* Custom Header Bar */
        .header-bar {
            background-color: #204855; /* ASI Dark Teal */
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            color: white;
            text-align: left; /* Changed from center to left for flex layout */
            background-image: linear-gradient(135deg, #204855 0%, #325A67 100%);
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .header-bar img {
            height: 80px;
            width: auto;
            border-radius: 4px;
        }
        .header-content {
            flex: 1;
        }
        .header-bar h1 {
            color: white !important;
            margin-bottom: 0.5rem;
            margin-top: 0;
        }
        .header-bar p {
            font-size: 1.1rem;
            opacity: 0.95;
            color: white;
            margin: 0;
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

NC_SPLIT_COLUMNS = {"NC number", "Client name", "Indicator", "Grade", "Status", "Issue Date", "Closed date"}


def expand_nc_rows_from_single(single_entry, column_names, filename):
    """Best-effort split of aggregated NC data into multiple rows."""
    if not single_entry:
        return []
    processed = {}
    max_len = 0
    # Store rich metadata to re-attach later
    rich_metadata = {} 
    
    for column in column_names:
        val_obj = single_entry.get(column, "")
        # Handle rich object or legacy string
        if isinstance(val_obj, dict) and "answer" in val_obj:
            raw_val = str(val_obj["answer"] or "")
            rich_metadata[column] = val_obj # Keep the full object
        else:
            raw_val = str(val_obj or "")
            rich_metadata[column] = None

        if column in NC_SPLIT_COLUMNS:
            segments = [seg.strip() for seg in re.split(r";|\n", raw_val) if seg.strip()]
        else:
            segments = [raw_val.strip()] if raw_val.strip() else [""]
        if not segments:
            segments = [""]
        processed[column] = segments
        max_len = max(max_len, len(segments))

    max_len = max(max_len, 1)
    rows = []
    for idx in range(max_len):
        row = {"filename": filename}
        for column in column_names:
            values = processed[column]
            answer_val = values[idx] if idx < len(values) else values[-1]
            
            # If we had rich metadata, re-wrap the answer with it
            meta = rich_metadata.get(column)
            if meta:
                # Create a shallow copy to avoid reference issues, update answer
                new_obj = meta.copy()
                new_obj['answer'] = answer_val
                row[column] = new_obj
            else:
                row[column] = answer_val
        rows.append(row)
    return rows

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
        "instruction_array": (
            "Focus strictly on Section 4.3.1 'Non-Conformities Identified during this Audit'. "
            "Return a JSON array where each element represents a single non-conformity entry."
        ),
        "instruction_fallback": (
            "Focus strictly on Section 4.3.1 'Non-Conformities Identified during this Audit'. "
            "Return a JSON object with the requested fields. If multiple NCs exist, list them all in order, "
            "combining values within each field separated by ';'."
        ),
        "file_name": "non_conformities_current_audit.csv"
    },
    {
        "key": "previous_nc",
        "title": "4.3.2 Non-Conformities Identified during the last ASA",
        "instruction_array": (
            "Focus strictly on Section 4.3.2 'Non-Conformity Identified during the last ASA'. "
            "Return a JSON array where each element represents a single non-conformity entry."
        ),
        "instruction_fallback": (
            "Focus strictly on Section 4.3.2 'Non-Conformity Identified during the last ASA'. "
            "Return a JSON object with the requested fields. If multiple NCs exist, list them all in order, "
            "combining values within each field separated by ';'."
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


def generate_gemini_schema(schema_dict, as_array=False):
    """Converts user schema to Gemini response schema with source verification."""
    properties = {}
    required = []
    for col_name, question in schema_dict.items():
        properties[col_name] = {
            "type": "OBJECT",
            "properties": {
                "answer": {"type": "STRING", "description": question},
                "source_quote": {"type": "STRING", "description": "The exact substring from the text that supports this answer."},
                "page_number": {"type": "INTEGER", "description": "The page number (1-indexed) where the quote is found."}
            },
            "required": ["answer", "source_quote", "page_number"]
        }
        required.append(col_name)
    
    base_obj = {
        "type": "OBJECT",
        "properties": properties,
        "required": required
    }
    
    if as_array:
        return {
            "type": "ARRAY",
            "items": base_obj
        }
    return base_obj


def analyze_document_with_gemini(filename, file_path, schema_dict, api_key, model_name, extra_instruction=None, expect_list=False):
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
        schema = generate_gemini_schema(schema_dict, as_array=expect_list)
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        )

        prompt = "Extract the information from the provided document according to the JSON schema. For each field, provide the Answer, the Exact Source Quote from the text, and the Page Number."
        if extra_instruction:
            prompt += f" {extra_instruction}"

        response = model.generate_content([uploaded_file, prompt])
        response_payload = _clean_json_payload(getattr(response, "text", "") or "")
        data = json.loads(response_payload or "[]")
        
        # Add filename to the root object (for list or dict)
        if expect_list:
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                data = []
            for row in data:
                if isinstance(row, dict):
                    row['filename'] = filename
            return data
            
        if not isinstance(data, dict):
            # Fallback for unexpected structure
            data = {}
        
        # Ensure we keep the structure for Source Verification, but also flatten for main table if needed later.
        # Ideally, we return the rich object as is.
        data['filename'] = filename
        return data

    except Exception as e:
        # Return error dict to keep table alignment
        err_row = {key: "Extraction Error" for key in schema_dict.keys()}
        err_row['filename'] = filename
        st.error(f"Gemini extraction failed for {filename}: {e}")
        if expect_list:
            return [err_row]
        return err_row

# ==========================================
# APP UI
# ==========================================

logo_b64 = get_image_base64("logo.png")
logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="fixed-logo">' if logo_b64 else ""

st.markdown(f"{logo_html}", unsafe_allow_html=True)

st.markdown("""
    <div class="header-bar">
        <div class="header-content">
            <h1>ASI Audit Intelligence Platform</h1>
            <p>Automated analysis of audit reports with structured data export for compliance, reporting, and operational insight.</p>
        </div>
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

@st.dialog("Source Verification", width="large")
def show_source_verification(row_data, schema_dict_local, title):
    st.markdown(f"### {title}")
    
    # Find the uploaded file to display
    target_filename = row_data.get("filename")
    pdf_file_buffer = None
    if uploaded_files:
        for f in uploaded_files:
            if f.name == target_filename:
                pdf_file_buffer = f
                break
    
    if not pdf_file_buffer:
        st.error(f"Could not find PDF file: {target_filename}")
        return

    # Layout: Left column for fields, Right column for PDF
    col1, col2 = st.columns([1, 1])
    
    selected_field = None
    
    with col1:
        st.markdown("#### Extracted Fields")
        # Show fields as selectable pills or radio buttons
        # Filter out metadata keys like filename
        field_keys = [k for k in row_data.keys() if k != "filename"]
        
        # Use a radio button for selection, simpler than pills for now
        selected_field_key = st.radio(
            "Select a field to verify:",
            field_keys,
            format_func=lambda x: x
        )
        
        if selected_field_key:
            st.divider()
            val = row_data[selected_field_key]
            if isinstance(val, dict):
                ans = val.get("answer", "N/A")
                quote = val.get("source_quote", "N/A")
                page = val.get("page_number", 1)
                st.markdown(f"**Answer:** {ans}")
                st.info(f"**Source Quote:** \"{quote}\"")
                st.markdown(f"**Found on Page:** {page}")
                
                selected_field = {
                    "page": page,
                    "quote": quote
                }
            else:
                st.markdown(f"**Value:** {val}")
                st.warning("No source metadata available for this field.")
                selected_field = {"page": 1, "quote": ""}

    with col2:
        if pdf_file_buffer:
            # Prepare PDF data
            binary_data = pdf_file_buffer.getvalue()
            
            # Determine page to show
            page_num = selected_field["page"] if selected_field else 1
            quote_text = selected_field["quote"] if selected_field else ""
            
            # Extract coordinates for highlighting
            annotations = []
            if quote_text and page_num:
                try:
                    with pdfplumber.open(BytesIO(binary_data)) as pdf:
                        # pdfplumber pages are 0-indexed, Gemini is 1-indexed
                        if 0 <= page_num - 1 < len(pdf.pages):
                            page = pdf.pages[page_num - 1]
                            # Search for the quote
                            words = page.search(quote_text)
                            if words:
                                # Start with the first match
                                # structure: [{'x0': ..., 'top': ..., 'x1': ..., 'bottom': ...}, ...]
                                # pdf_viewer expects annotations in a specific format or we can overlay basic rectangles
                                # The streamlit-pdf-viewer expects `annotations` as a list of dictionaries.
                                # Each dictionary should have: page, x, y, width, height, color
                                # Note: pdfplumber gives (x0, top, x1, bottom) where (0,0) is top-left usually.
                                
                                for w in words:
                                    annotations.append({
                                        "page": page_num,
                                        "x": w["x0"],
                                        "y": w["top"],
                                        "width": w["x1"] - w["x0"],
                                        "height": w["bottom"] - w["top"],
                                        "color": "yellow",
                                        "opacity": 0.3
                                    })
                except Exception as e:
                    st.warning(f"Could not highlight text: {e}")

            # Use streamlit-pdf-viewer
            st.markdown(f"**Viewing Page: {page_num}**")
            pdf_viewer(
                input=binary_data, 
                width=700, 
                height=800, 
                pages_to_render=[page_num],
                annotations=annotations if annotations else None
            )

def flatten_data(rich_data):
    """Flattens a list of rich objects into a simple dict list for display."""
    flat_data = []
    for item in rich_data:
        flat_row = {}
        for k, v in item.items():
            if isinstance(v, dict) and "answer" in v:
                flat_row[k] = v["answer"]
            else:
                flat_row[k] = v
        flat_data.append(flat_row)
    return flat_data


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

        @st.dialog("Source Verification", width="large")
        def show_source_verification(row_data, schema_dict, title):
            st.markdown(f"### {title}")
            
            # Find the uploaded file to display
            target_filename = row_data.get("filename")
            pdf_file_buffer = None
            for f in uploaded_files:
                if f.name == target_filename:
                    pdf_file_buffer = f
                    break
            
            if not pdf_file_buffer:
                st.error(f"Could not find PDF file: {target_filename}")
                return

            # Layout: Left column for fields, Right column for PDF
            col1, col2 = st.columns([1, 1])
            
            selected_field = None
            
            with col1:
                st.markdown("#### Extracted Fields")
                # Show fields as selectable pills or radio buttons
                # Filter out metadata keys like filename
                field_keys = [k for k in row_data.keys() if k != "filename"]
                
                # Use a radio button for selection, simpler than pills for now
                selected_field_key = st.radio(
                    "Select a field to verify:",
                    field_keys,
                    format_func=lambda x: x
                )
                
                if selected_field_key:
                    st.divider()
                    val = row_data[selected_field_key]
                    if isinstance(val, dict):
                        ans = val.get("answer", "N/A")
                        quote = val.get("source_quote", "N/A")
                        page = val.get("page_number", 1)
                        st.markdown(f"**Answer:** {ans}")
                        st.info(f"**Source Quote:** \"{quote}\"")
                        st.markdown(f"**Found on Page:** {page}")
                        
                        selected_field = {
                            "page": page,
                            "quote": quote
                        }
                    else:
                        st.markdown(f"**Value:** {val}")
                        st.warning("No source metadata available for this field.")
                        selected_field = {"page": 1, "quote": ""}

            with col2:
                if pdf_file_buffer:
                    base64_pdf = base64.b64encode(pdf_file_buffer.getvalue()).decode('utf-8')
                    
                    # Construct PDF URL with page fragment
                    # Browser PDF viewers usually support #page=N
                    page_num = selected_field["page"] if selected_field else 1
                    quote_text = selected_field["quote"] if selected_field else ""
                    
                    # Basic PDF embedding
                    # We try to use the browser's native viewer via iframe
                    # Adding #page=N to data URI might not work in all browsers, 
                    # but usually works for simple navigation. 
                    # Text fragment #:~:text= is supported in Chrome/Edge.
                    
                    # Clean quote for URL fragment (simple encoding)
                    import urllib.parse
                    quote_fragment = f"#:~:text={urllib.parse.quote(quote_text)}" if quote_text else ""
                    pdf_src = f"data:application/pdf;base64,{base64_pdf}#page={page_num}{quote_fragment}"
                    
                    st.markdown(f'<iframe src="{pdf_src}" width="100%" height="600px"></iframe>', unsafe_allow_html=True)


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
            
            # Clear previous results from session state
            st.session_state.rich_results_main = []
            st.session_state.schema_dict = schema_dict # Store schema for later use
            st.session_state.nc_schema_dicts = nc_schema_dicts if scan_nc else {} # Store nc schemas
            
            # Initialize NC results in session state
            if scan_nc:
                for section in NC_SECTIONS:
                    st.session_state[f"rich_results_{section['key']}"] = []

            for i, pdf_file in enumerate(uploaded_files):
                status_text.text(f"Processing {pdf_file.name}...")
                
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    # 1. Process Main Schema
                    extracted_data = analyze_document_with_gemini(
                        pdf_file.name, tmp_file_path, schema_dict, API_KEY, MODEL_NAME
                    )
                    results.append(extracted_data)
                    # Update session state immediately
                    st.session_state.rich_results_main = results
                    # Force rerun to update table? No, we will rely on reactiveness or manual rerun if needed.
                    # Actually, we can just rely on the layout below to render it.

                    # 2. Process Non-Conformities
                    if scan_nc:
                        for section in NC_SECTIONS:
                            section_schema = {
                                col: f"{desc} (Context: {section['title']})."
                                for col, desc in nc_schema_dicts[section["key"]].items()
                            }
                            try:
                                # Attempt array extraction first
                                section_data = analyze_document_with_gemini(
                                    pdf_file.name,
                                    tmp_file_path,
                                    section_schema,
                                    API_KEY,
                                    MODEL_NAME,
                                    extra_instruction=section["instruction_array"],
                                    expect_list=True
                                )
                                # Fallback if empty or failed
                                if not section_data:
                                    fallback_single = analyze_document_with_gemini(
                                        pdf_file.name,
                                        tmp_file_path,
                                        section_schema,
                                        API_KEY,
                                        MODEL_NAME,
                                        extra_instruction=section["instruction_fallback"],
                                        expect_list=False
                                    )
                                    section_data = expand_nc_rows_from_single(
                                        fallback_single,
                                        list(nc_schema_dicts[section["key"]].keys()),
                                        pdf_file.name
                                    )
                                
                                # Append and update session state
                                nc_results[section["key"]].extend(section_data or [])
                                st.session_state[f"rich_results_{section['key']}"] = nc_results[section["key"]]
                                
                            except Exception as nc_err:
                                st.error(f"Error processing non-conformities for {pdf_file.name} ({section['title']}): {nc_err}")
                                error_row = {
                                    col: "Extraction Error" for col in nc_schema_dicts[section["key"]].keys()
                                }
                                error_row["filename"] = pdf_file.name
                                nc_results[section["key"]].append(error_row)
                                st.session_state[f"rich_results_{section['key']}"] = nc_results[section["key"]]

                except Exception as e:
                    st.error(f"Error processing {pdf_file.name}: {e}")
                    error_row = {key: "Extraction Error" for key in schema_dict.keys()}
                    error_row['filename'] = pdf_file.name
                    results.append(error_row)
                    st.session_state.rich_results_main = results
                finally:
                    # Clean up the temporary file
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                time.sleep(0.5)

            status_text.text("Done!")
            st.rerun() # Rerun to render the tables in the main flow

        process_documents()

# ==========================================
# RESULT DISPLAY (runs on every script rerun)
# ==========================================

# 1. Main Results
if "rich_results_main" in st.session_state and st.session_state.rich_results_main:
    st.divider()
    st.subheader("3. Extracted Data")
    
    results = st.session_state.rich_results_main
    flat_results = flatten_data(results)
    df = pd.DataFrame(flat_results)
    
    # Ensure columns match schema
    current_schema = st.session_state.get("schema_dict", {})
    cols = ['filename'] + [k for k in current_schema.keys() if k != 'filename']
    for col in cols:
        if col not in df.columns:
            df[col] = "N/A"
    df = df[cols] if cols else df
    
    event = st.dataframe(
        df, 
        width='stretch', 
        on_select="rerun", 
        selection_mode="single-row",
        key="main_table_df"
    )
    
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        # Pass the schema dict stored in session
        show_source_verification(results[selected_idx], st.session_state.get("schema_dict", {}), "Main Extraction")

    # Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="extracted_data.csv",
        mime="text/csv",
    )

# 2. Non-Conformity Results
if scan_nc_checked := scan_nc: # check the current widget state
    st.divider()
    st.subheader("4. Non-Conformities")
    
    # Retrieve schemas from session if available
    nc_schemas = st.session_state.get("nc_schema_dicts", {})
    
    for section in NC_SECTIONS:
        key = section["key"]
        res_key = f"rich_results_{key}"
        if res_key in st.session_state and st.session_state[res_key]:
            st.markdown(f"**{section['title']}**")
            
            nc_rows = st.session_state[res_key]
            flat_nc = flatten_data(nc_rows)
            df_nc = pd.DataFrame(flat_nc)
            
            # Ensure columns
            sec_schema = nc_schemas.get(key, {})
            cols = ['filename'] + list(sec_schema.keys())
            for col in cols:
                if col not in df_nc.columns:
                    df_nc[col] = "N/A"
            df_nc = df_nc[cols] if cols else df_nc
            
            event_nc = st.dataframe(
                df_nc, 
                width='stretch', 
                on_select="rerun", 
                selection_mode="single-row",
                key=f"nc_table_{key}"
            )
            
            if event_nc.selection.rows:
                sel_idx = event_nc.selection.rows[0]
                show_source_verification(nc_rows[sel_idx], sec_schema, f"Non-Conformity: {key}")

            csv_nc = df_nc.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download CSV - {section['title']}",
                data=csv_nc,
                file_name=section["file_name"],
                mime="text/csv",
            )
