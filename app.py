import streamlit as st
from PIL import Image
import time
import hashlib 
import database_helper as db # Replaces the old direct MySQL calls
import streamlit.components.v1 as components
import pandas as pd 
import io
from fpdf import FPDF 
import tempfile 
import os 

# Initialize the SQLite database file on startup
db.init_db()

# Import ML logic from your provided module
from ml_logic import (
    get_sample_options, 
    display_training_structure, 
    load_model_real, 
    get_real_prediction, 
    get_disease_from_sample,
    DISEASE_MODELS 
)

# --- Password Utilities ---
def hash_password(password):
    """Hashes a password using SHA-256 for secure storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, provided_password):
    """Verifies a provided password against the stored hash."""
    return stored_hash == hash_password(provided_password)

# --- PDF Report Generation Class ---
def create_pdf_report(analysis_result, patient_name, sample_type, uploaded_image, grad_cam_image):
    """Generates a detailed diagnosis report in PDF format using FPDF."""
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Pathoscope AI Diagnosis Report', 0, 1, 'C')
            self.line(10, 20, 200, 20)
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', '', 11)
    
    # Report Metadata
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Report Generated On: {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'L')
    pdf.cell(0, 5, f"Analyst: {st.session_state.get('full_name', 'N/A')} (ID: {st.session_state.get('user_id', 'N/A')})", 0, 1, 'L')
    pdf.ln(5)
    
    # 1. Patient and Sample Details
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, '1. Patient and Sample Details', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(60, 7, 'Patient Name:', 1, 0, 'L', 1)
    pdf.cell(130, 7, patient_name, 1, 1, 'L')
    pdf.cell(60, 7, 'Sample Type:', 1, 0, 'L', 1)
    pdf.cell(130, 7, sample_type, 1, 1, 'L')
    pdf.cell(60, 7, 'Disease Tested:', 1, 0, 'L', 1)
    pdf.cell(130, 7, analysis_result['disease'], 1, 1, 'L')
    pdf.ln(5)

    # 2. Diagnosis Result
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, '2. Diagnosis Result', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)

    res_upper = analysis_result['result_status'].upper()
    result_color = (255, 0, 0) if any(x in res_upper for x in ['POSITIVE', 'MALIGNANT', 'PARASITIZED']) else (0, 128, 0)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(60, 7, 'Result Status:', 1, 0, 'L', 1)
    pdf.set_text_color(*result_color)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(130, 7, analysis_result['result_status'], 1, 1, 'L')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    
    pdf.cell(60, 7, 'Confidence Score:', 1, 0, 'L', 1)
    pdf.cell(130, 7, f"{analysis_result['percentage']:.2f}%", 1, 1, 'L')
    pdf.cell(60, 7, 'Model Used:', 1, 0, 'L', 1)
    pdf.cell(130, 7, DISEASE_MODELS[analysis_result['disease']]['model_file'], 1, 1, 'L')
    pdf.ln(8)

    # 3. Visual Analysis
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, '3. Visual Analysis', 0, 1, 'L')
    
    def add_image_to_pdf(pdf_doc, image_obj, title, caption):
        pdf_doc.set_font('Arial', 'BU', 11)
        pdf_doc.cell(0, 7, title, 0, 1, 'L')
        pdf_doc.set_font('Arial', 'I', 9)
        pdf_doc.multi_cell(0, 5, caption, 0, 'L')
        pdf_doc.ln(2)
        if image_obj:
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_file_path = tmp.name
                    image_obj.save(temp_file_path, format="PNG") 
                pdf_doc.image(temp_file_path, w=90) 
            except Exception as e:
                pdf_doc.set_font('Arial', 'I', 10)
                pdf_doc.cell(0, 5, f'Error embedding image: {e}', 0, 1, 'L')
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            pdf_doc.ln(5)
        else:
            pdf_doc.set_font('Arial', 'I', 10)
            pdf_doc.cell(0, 5, 'Image not available.', 0, 1, 'L')
            pdf_doc.ln(5)

    add_image_to_pdf(pdf, uploaded_image, 'A. Original Sample Image', 'The raw microscopic image uploaded for analysis.')
    add_image_to_pdf(pdf, grad_cam_image, 'B. Grad-CAM Visualization', 'Regions indicate model focus.')
    
    if pdf.get_y() > 250:
        pdf.add_page()
    else:
        pdf.set_y(pdf.get_y() + 10)
        
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, '4. Disclaimer', 0, 1, 'L')
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(150, 0, 0)
    disclaimer = ("This report is for informational purposes only and is based on an automated AI analysis model. "
                  "It is NOT a substitute for professional medical diagnosis.")
    pdf.multi_cell(0, 6, disclaimer, 0, 'J')
    
    return pdf.output(dest='S').encode('latin1')

# --- Page: Login/Signup ---
def page_login():
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/DNA_sequence_by_Sanger_method.png/440px-DNA_sequence_by_Sanger_method.png", width=100)
    st.header("Pathoscope AI Login")
    st.info("Log in or use **anon/anon** to explore.")

    choice = st.selectbox("Action", ["Login", "Sign Up"], key="auth_choice")

    if choice == "Login":
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username == "anon" and password == "anon":
                    st.session_state.update({'logged_in': True, 'user_id': 9999, 'full_name': "Anonymous User"})
                    st.rerun()
                
                user = db.authenticate_user(username, password)
                if user:
                    st.session_state.update({'logged_in': True, 'user_id': user['id'], 'full_name': user['full_name']})
                    st.rerun()
                else:
                    st.error("Invalid Username or Password.")

    elif choice == "Sign Up":
        with st.form("signup_form"):
            new_full_name = st.text_input("Full Name", key="signup_name")
            new_username = st.text_input("New Username", key="signup_user")
            new_password = st.text_input("New Password", type="password", key="signup_pass")
            submitted = st.form_submit_button("Sign Up", use_container_width=True)
            if submitted:
                if new_username and new_password and new_full_name:
                    if db.save_user(new_username, new_password, new_full_name):
                        st.success("Signup successful! You can now log in.")
                else:
                    st.warning("Please fill in all fields.")

# --- Page: Diagnosis ---
def page_diagnosis():
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/DNA_sequence_by_Sanger_method.png/440px-DNA_sequence_by_Sanger_method.png", width=100)
    st.title("Smart Microscope Diagnosis System")
    display_training_structure()
    
    st.subheader("Patient & Sample Details")
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name:", value=st.session_state.get('full_name', ''))
    with col2:
        sample_type = st.selectbox("Sample Type", get_sample_options())
    
    disease_tested = get_disease_from_sample(sample_type)
    model = load_model_real(disease_tested) 

    st.subheader("Choose Image Source")
    input_method = st.radio("Select Input Method:", ("Upload Image from Device", "Capture Live via Microscope/Camera"))
    
    uploaded_file = st.file_uploader("Upload microscope image", type=["png", "jpg", "jpeg"]) if "Upload" in input_method else st.camera_input("Take Picture")
        
    analyze_button = st.button("🔬 Analyze Sample", use_container_width=True, disabled=(uploaded_file is None or not patient_name))
    
    if analyze_button and uploaded_file and patient_name:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.session_state.update({'uploaded_image': image, 'patient_name_report': patient_name, 'sample_type_report': sample_type})
            
            with st.spinner(f"Analyzing Image using {DISEASE_MODELS[disease_tested]['model_file']}..."):
                analysis_result = get_real_prediction(model, image, disease_tested)
            
            st.session_state['analysis_result'] = analysis_result
            st.header("Diagnosis Result")
            
            res_status = analysis_result['result_status']
            conf = analysis_result['percentage']
            
            # Use specific class checks for styling
            is_pos = any(x in res_status.upper() for x in ['POSITIVE', 'MALIGNANT', 'PARASITIZED'])
            if is_pos:
                st.error(f"Result: **POSITIVE INDICATOR** for {disease_tested} ({res_status}) ({conf:.2f}%)")
            else:
                st.success(f"Result: **NEGATIVE INDICATOR** for {disease_tested} ({res_status}) ({conf:.2f}%)")
            
            st.subheader("Grad-CAM Visualization")
            st.image(analysis_result['grad_cam_image'], caption="AI Focus Regions", use_container_width=True)
            st.session_state['grad_cam_image'] = analysis_result['grad_cam_image']
            
            # Save to SQLite Database
            if st.session_state['user_id'] != 9999:
                db.save_diagnosis(
                    st.session_state['user_id'], patient_name, sample_type,
                    disease_tested, res_status, analysis_result['confidence_decimal']
                )
                st.success("Diagnosis record saved to your history.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")

    # Report Download Section
    if 'analysis_result' in st.session_state:
        st.markdown("---")
        st.subheader("Download Report")
        try:
            pdf_bytes = create_pdf_report(
                st.session_state['analysis_result'], 
                st.session_state.get('patient_name_report', 'N/A'), 
                st.session_state.get('sample_type_report', 'N/A'), 
                st.session_state.get('uploaded_image'), 
                st.session_state.get('grad_cam_image')
            )
            st.download_button(label="⬇️ Download Detailed Report (PDF)", data=pdf_bytes, 
                             file_name=f"Pathoscope_Report_{st.session_state.get('patient_name_report')}.pdf", 
                             mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"PDF Error: {e}")

# --- Page: History ---
def page_history():
    st.title("📚 Diagnosis History")
    if st.session_state['user_id'] == 9999:
        st.warning("History is not saved for anonymous users.")
        return

    st.subheader("Search Records")
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input("Enter patient name or disease", key="voice_search_input")
    with col2:
        st.write(" ")
        voice_js = """
        <script>
        function startDictation() {
            if (window.hasOwnProperty('webkitSpeechRecognition')) {
                var recognition = new webkitSpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = "en-US";
                recognition.start();
                recognition.onresult = function(e) {
                    var voiceText = e.results[0][0].transcript;
                    recognition.stop();
                    const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                    let targetInput = null;
                    for (let input of inputs) {
                        if (input.placeholder === "Enter patient name or disease") {
                            targetInput = input; break;
                        }
                    }
                    if (targetInput) {
                        const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                        setter.call(targetInput, voiceText);
                        targetInput.dispatchEvent(new Event('input', { bubbles: true }));
                        targetInput.focus();
                        setTimeout(() => { targetInput.blur(); }, 100);
                    }
                };
            }
        }
        </script>
        <button onclick="startDictation()" style="background:#ff4b4b; color:white; border:none; padding:10px; border-radius:8px; cursor:pointer; width:100%; height:45px; font-weight:bold;">🎤 Click to Speak</button>
        """
        components.html(voice_js, height=45)

    records = db.fetch_history(st.session_state['user_id'])
    if records:
        df = pd.DataFrame(records)
        # Rename for display
        df = df.rename(columns={
            'diagnosis_date': 'Date', 'patient_name': 'Patient', 
            'sample_type': 'Sample', 'disease_tested': 'Disease', 
            'result_status': 'Result', 'confidence_score': 'Confidence'
        })
        # Filtering
        if search_query:
            df = df[df['Patient'].str.contains(search_query, case=False) | df['Disease'].str.contains(search_query, case=False)]
        
        st.subheader(f"Total Records: {len(df)}")
        st.dataframe(df[["Date", "Patient", "Sample", "Disease", "Result", "Confidence"]], use_container_width=True, hide_index=True)
    else:
        st.warning("No records found.")

# --- Main App Entry ---
def main():
    st.set_page_config(page_title="Pathoscope AI", page_icon="🔬", layout="wide")
    if 'logged_in' not in st.session_state:
        st.session_state.update({'logged_in': False, 'page': 'Login'})

    if not st.session_state['logged_in']:
        page_login()
    else:
        st.sidebar.title("Pathoscope AI")
        st.sidebar.markdown(f"**Welcome, {st.session_state['full_name']}**")
        app_page = st.sidebar.radio("Navigate", ["Diagnosis", "History"])
        if st.sidebar.button("🔒 Logout", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

        if app_page == "Diagnosis": page_diagnosis()
        else: page_history()

if __name__ == '__main__':
    main()
