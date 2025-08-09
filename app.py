import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import io
from datetime import datetime
import time
from cv2con import extract_text_from_pdf, extract_text_from_docx
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

# Page configuration
st.set_page_config(
    page_title="CV2CSV - AI Resume Data Extractor",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'custom_fields' not in st.session_state:
    st.session_state.custom_fields = []

generation_config = {"temperature": 0.02}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name='gemini-2.0-flash',
                                   generation_config=generation_config,)

def build_dynamic_prompt(custom_fields):
    """Build prompt with custom fields only"""
    
    if not custom_fields:
        st.warning("⚠️ Please add fields in the sidebar before processing resumes.")
        return {}, [], []
    
    # Build field descriptions for AI
    field_descriptions = []
    for field in custom_fields:
        field_descriptions.append(f"- {field['field_name']}: {field['description']}")
    
    # Instructions
    special_instructions = [
        "1. If any field is not found, use 'Not Found' as the value",
        "2. Extract only factual information present in the resume",
        "3. Be precise and consistent with data formats",
        "4. For numeric fields, extract only numbers when possible",
        "5. For date fields, use standard formats (YYYY or MM/YYYY)"
    ]
    
    return custom_fields, field_descriptions, special_instructions

def parse_resume_with_gemma(model, resume_text, custom_fields):
    """Parse resume using Gemma model with dynamic fields"""
    
    if not custom_fields:
        return None
    
    all_fields, field_descriptions, special_instructions = build_dynamic_prompt(custom_fields)
    
    prompt = f"""
Extract the following information from this resume text and return it as a JSON object:

**Required Fields:**
{chr(10).join(field_descriptions)}

**Instructions:**
{chr(10).join(special_instructions)}

**Output Format:**
Return only a valid JSON object with the exact field names specified above.

**Resume Text:**
{resume_text}

**Response:**
"""
    
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        # Clean JSON response
        if json_text.startswith('```json'):
            json_text = json_text[7:-3]
        elif json_text.startswith('```'):
            json_text = json_text[3:-3]
            
        return json.loads(json_text)
    except Exception as e:
        st.error(f"Error parsing resume with Gemma: {e}")
        return None

def process_uploaded_files(model, uploaded_files, custom_fields):
    """Process all uploaded resume files"""
    
    if not custom_fields:
        st.error("❌ Please add fields in the sidebar before processing.")
        return []
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Extract text based on file type
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            resume_text = extract_text_from_pdf(uploaded_file)
        elif file_extension in ['docx', 'doc']:
            resume_text = extract_text_from_docx(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue
            
        if resume_text:
            # Parse with Gemma
            parsed_data = parse_resume_with_gemma(model, resume_text, custom_fields)
            
            if parsed_data:
                parsed_data['file_name'] = uploaded_file.name
                parsed_data['processed_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                results.append(parsed_data)
                st.success(f"✅ Processed: {uploaded_file.name}")
            else:
                st.error(f"❌ Failed to process: {uploaded_file.name}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    status_text.text("Processing complete!")
    return results

def render_sidebar():
    """Render the sidebar for field management"""
    
    st.sidebar.title("📋 Field Configuration")
    st.sidebar.markdown("Configure fields to extract from resumes")
    
    # Add new field section
    st.sidebar.header("➕ Add New Field")
    
    with st.sidebar.form("add_field_form", clear_on_submit=True):
        field_name = st.text_input(
            "Field Name *",
            placeholder="e.g., candidate_name",
            help="Internal field name (snake_case, no spaces)"
        )
        
        description = st.text_area(
            "Description *",
            placeholder="e.g., Full name of the candidate",
            help="Describe what this field should extract from resume"
        )
        
        column_name = st.text_input(
            "Column Name *",
            placeholder="e.g., Full Name",
            help="Display name for CSV column header"
        )
        
        submitted = st.form_submit_button("Add Field", type="primary", use_container_width=True)
        
        if submitted:
            if field_name and description and column_name:
                new_field = {
                    'field_name': field_name.strip(),
                    'description': description.strip(),
                    'column_name': column_name.strip()
                }
                
                # Check if field already exists
                existing_names = [f['field_name'] for f in st.session_state.custom_fields]
                if field_name.lower() not in existing_names:
                    st.session_state.custom_fields.append(new_field)
                    st.sidebar.success(f"✅ Added: {column_name}")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Field name already exists!")
        else:
            st.sidebar.error("❌ All fields are required!")
    
    # Display current fields
    st.sidebar.header("📝 Current Fields")
    
    if st.session_state.custom_fields:
        st.sidebar.info(f"Total Fields: {len(st.session_state.custom_fields)}")
        
        for i, field in enumerate(st.session_state.custom_fields):
            with st.sidebar.container():
                col1, col2 = st.sidebar.columns([3, 1])
                
                with col1:
                    st.write(f"**{field['column_name']}**")
                    st.caption(f"Field: `{field['field_name']}`")
                    st.caption(f"Desc: {field['description'][:40]}...")
                
                with col2:
                    if st.button("🗑️", key=f"del_{i}", help="Delete field"):
                        st.session_state.custom_fields.pop(i)
                        st.rerun()
                
                st.sidebar.divider()
        
        # Clear all fields button
        if st.sidebar.button("🗑️ Clear All Fields", type="secondary", use_container_width=True):
            st.session_state.custom_fields = []
            st.session_state.processed_data = []
            st.session_state.processing_complete = False
            st.rerun()
    else:
        st.sidebar.info("No fields configured yet.")
        st.sidebar.markdown("""
        **Quick Start:**
        1. Add field name (e.g., `full_name`)
        2. Add description for AI extraction
        3. Add column name for CSV header
        4. Click 'Add Field'
        """)
    
    # Predefined templates
    st.sidebar.header("🚀 Quick Templates")
    
    if st.sidebar.button("📄 Basic Resume Fields", use_container_width=True):
        basic_fields = [
            {"field_name": "full_name", "description": "Full name of the candidate", "column_name": "Full Name"},
            {"field_name": "email_address", "description": "Email address of the candidate", "column_name": "Email"},
            {"field_name": "phone_number", "description": "Phone number of the candidate", "column_name": "Phone"},
            {"field_name": "current_location", "description": "Current location or address", "column_name": "Location"},
            {"field_name": "total_experience", "description": "Total years of work experience", "column_name": "Experience (Years)"},
            {"field_name": "current_role", "description": "Current job title or position", "column_name": "Current Role"},
            {"field_name": "highest_education", "description": "Highest educational qualification", "column_name": "Education"},
            {"field_name": "key_skills", "description": "Technical and professional skills", "column_name": "Skills"},
        ]
        st.session_state.custom_fields = basic_fields
        st.rerun()
    
    if st.sidebar.button("🎓 Education Focus", use_container_width=True):
        education_fields = [
            {"field_name": "candidate_name", "description": "Full name of the candidate", "column_name": "Name"},
            {"field_name": "contact_email", "description": "Email address", "column_name": "Email"},
            {"field_name": "ug_degree", "description": "Undergraduate degree name", "column_name": "UG Degree"},
            {"field_name": "ug_college", "description": "Undergraduate college name", "column_name": "UG College"},
            {"field_name": "ug_year", "description": "Undergraduate graduation year", "column_name": "UG Year"},
            {"field_name": "pg_degree", "description": "Postgraduate degree name", "column_name": "PG Degree"},
            {"field_name": "pg_college", "description": "Postgraduate college name", "column_name": "PG College"},
            {"field_name": "pg_year", "description": "Postgraduate graduation year", "column_name": "PG Year"},
            {"field_name": "cgpa_percentage", "description": "CGPA or percentage mentioned", "column_name": "CGPA/Percentage"},
        ]
        st.session_state.custom_fields = education_fields
        st.rerun()
    
    if st.sidebar.button("💼 HR Screening", use_container_width=True):
        hr_fields = [
            {"field_name": "applicant_name", "description": "Full name of the applicant", "column_name": "Applicant Name"},
            {"field_name": "contact_number", "description": "Contact phone number", "column_name": "Contact Number"},
            {"field_name": "email_id", "description": "Email address", "column_name": "Email ID"},
            {"field_name": "work_experience", "description": "Total work experience in years", "column_name": "Work Experience"},
            {"field_name": "current_salary", "description": "Current salary or CTC", "column_name": "Current Salary"},
            {"field_name": "expected_salary", "description": "Expected salary", "column_name": "Expected Salary"},
            {"field_name": "notice_period", "description": "Current notice period", "column_name": "Notice Period"},
            {"field_name": "current_company", "description": "Current company name", "column_name": "Current Company"},
            {"field_name": "preferred_location", "description": "Preferred work location", "column_name": "Preferred Location"},
        ]
        st.session_state.custom_fields = hr_fields
        st.rerun()

def main():
    # Render sidebar first
    render_sidebar()
    
    # Main content
    st.title("📄 CV2CSV - AI Resume Data Extractor")
    st.caption("Upload resumes and extract structured data using AI")
    
    
    # Show current configuration
    if st.session_state.custom_fields:
        st.success(f"✅ {len(st.session_state.custom_fields)} fields configured")
        
        # Show field preview
        with st.expander("👀 Preview Fields to Extract"):
            col1, col2, col3 = st.columns(3)
            
            for i, field in enumerate(st.session_state.custom_fields):
                with [col1, col2, col3][i % 3]:
                    st.info(f"**{field['column_name']}**\n\n`{field['field_name']}`")
    else:
        st.warning("⚠️ Please configure fields in the sidebar before uploading resumes.")
    
    st.markdown("---")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, DOC",
            disabled=len(st.session_state.custom_fields) == 0
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files uploaded")
            
            # Display uploaded files
            with st.expander("View uploaded files"):
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size:,} bytes)")
    
    with col2:
        st.header("⚙️ Processing")
        
        # Show stats
        if st.session_state.custom_fields:
            st.metric("Fields to Extract", len(st.session_state.custom_fields))
        else:
            st.metric("Fields to Extract", 0)
        
        # Process button
        process_disabled = not uploaded_files or len(st.session_state.custom_fields) == 0
        
        if st.button("🚀 Process All Resumes", type="primary", disabled=process_disabled):
            st.session_state.processed_data = []
            st.session_state.processing_complete = False
            
            with st.spinner("Processing resumes..."):
                results = process_uploaded_files(model, uploaded_files, st.session_state.custom_fields)
                st.session_state.processed_data = results
                st.session_state.processing_complete = True
            
            if results:
                st.success(f"✅ Successfully processed {len(results)} resumes!")
            else:
                st.error("❌ No resumes were processed successfully.")
        
        if process_disabled and uploaded_files:
            st.info("Configure fields in sidebar first")
    
    # Display results
    if st.session_state.processing_complete and st.session_state.processed_data:
        st.markdown("---")
        st.header("📊 Extracted Data")
        
        # Create DataFrame
        df = pd.DataFrame(st.session_state.processed_data)
        
        # Create column mapping for display names
        column_mapping = {}
        for field in st.session_state.custom_fields:
            column_mapping[field['field_name']] = field['column_name']
        
        # Add meta columns
        column_mapping['file_name'] = 'File Name'
        column_mapping['processed_date'] = 'Processed Date'
        
        # Rename columns for display
        display_df = df.rename(columns=column_mapping)
        
        # Reorder columns: file_name first, then custom fields, then processed_date last
        column_order = ['File Name']
        for field in st.session_state.custom_fields:
            if field['column_name'] in display_df.columns:
                column_order.append(field['column_name'])
        column_order.append('Processed Date')
        
        # Filter existing columns
        existing_columns = [col for col in column_order if col in display_df.columns]
        display_df = display_df[existing_columns]
        
        # Display data with scrollable table
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download section
        st.header("💾 Download Results")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # CSV download with display column names
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"resume_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        
        with col2:
            # Excel download with display column names
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                display_df.to_excel(writer, sheet_name='Resume Data', index=False)
            
            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"resume_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            col3a, col3b = st.columns(2)
            with col3a:
                st.metric("Resumes Processed", len(st.session_state.processed_data))
            with col3b:
                st.metric("Fields Extracted", len(st.session_state.custom_fields))

if __name__ == "__main__":
    main()