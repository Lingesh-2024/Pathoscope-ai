import streamlit as st
from PIL import Image
import time
import hashlib 
import mysql.connector
from ml_logic import (
    get_sample_options, 
    display_training_structure, 
    load_model_real, 
    get_real_prediction, 
    get_disease_from_sample,
    DISEASE_MODELS 
)
import streamlit.components.v1 as components
import pandas as pd 
import io
from fpdf import FPDF 
import tempfile 
import os 

# --- UPDATED CONFIGURATION (AIVEN CLOUD READY) ---
# We now use Streamlit Secrets instead of hardcoding 'localhost'
def get_db_connection():
    """Establishes a connection to the Aiven MySQL database using secrets."""
    try:
        conn = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            port=st.secrets["mysql"]["port"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        return conn
    except Exception as e:
        # Display a warning if the DB connection fails
        st.warning(f"Database Connection Warning: {e}. Check your Streamlit Secrets.")
        return None

def init_db():
    """Automatically creates the tables in Aiven if they don't exist yet."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # 1. Create Users Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255) NOT NULL
                )
            """)
            # 2. Create Patient History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    patient_name VARCHAR(255),
                    sample_type VARCHAR(255),
                    disease_tested VARCHAR(255),
                    result_status VARCHAR(255),
                    confidence_score FLOAT,
                    image_path TEXT,
                    diagnosis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        except Exception as e:
            st.error(f"Table Creation Error: {e}")
        finally:
            cursor.close()
            conn.close()

# Initialize the database tables on app launch
if 'db_checked' not in st.session_state:
    init_db()
    st.session_state['db_checked'] = True

def hash_password(password):
    """Hashes a password using SHA-256 for secure storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, provided_password):
    """Verifies a provided password against the stored hash."""
    return stored_hash == hash_password(provided_password)

def save_user(username, password, full_name):
    """Saves a new user to the database."""
    conn = get_db_connection()
    if not conn: 
        st.error("Database connection failed. Cannot register user at this time.")
        return False
    
    try:
        cursor = conn.cursor()
        # Check if username exists
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            st.warning("Username already exists. Please choose a different one.")
            return False
        
        password_hash = hash_password(password)
        query = "INSERT INTO users (username, password_hash, full_name) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, password_hash, full_name))
        conn.commit()
        st.success("Signup successful! You can now log in.")
        return True
    except Exception as e:
        st.error(f"Signup error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def authenticate_user(username, password):
    """Authenticates a user and returns their ID and full name."""
    conn = get_db_connection()
    
    # Fallback mechanism if database is offline
    if not conn: 
        if username == "admin" and password == "admin123":
            st.info("Emergency Login: Using offline admin credentials.")
            return 9999, "Offline Administrator"
        return None, None

    try:
        # Use dictionary=True so we can access columns by name (user['password_hash'])
        cursor = conn.cursor(dictionary=True)
        query = "SELECT id, password_hash, full_name FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user = cursor.fetchone()
        
        if user and verify_password(user['password_hash'], password):
            return user['id'], user['full_name']
        return None, None
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return None, None
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()


def save_diagnosis(user_id, patient_name, sample_type, disease_tested, result_status, confidence_score, image_path="N/A"):
    """Saves a diagnosis record to the database."""
    conn = get_db_connection()
    if not conn:
        # We return True here to avoid blocking the user experience if the DB is momentarily down
        # In a production app, you might cache this locally instead.
        return True 
    
    try:
        cursor = conn.cursor()
        # The schema assumes patient_history has: id, user_id, patient_name, sample_type, 
        # disease_tested, result_status, confidence_score, image_path, and diagnosis_date.
        query = """
        INSERT INTO patient_history 
        (user_id, patient_name, sample_type, disease_tested, result_status, confidence_score, image_path) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (user_id, patient_name, sample_type, disease_tested, result_status, confidence_score, image_path))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving diagnosis history: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()

def fetch_history(user_id):
    """Fetches all diagnosis records for the logged-in user."""
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        # Using dictionary=True makes it much easier to display records in a Streamlit table/dataframe
        cursor = conn.cursor(dictionary=True)
        query = """
        SELECT patient_name, sample_type, disease_tested, result_status, confidence_score, diagnosis_date 
        FROM patient_history 
        WHERE user_id = %s
        ORDER BY diagnosis_date DESC
        """
        cursor.execute(query, (user_id,))
        records = cursor.fetchall()
        return records
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return []
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()
        
import time
import os
import tempfile
from fpdf import FPDF
import streamlit as st
from PIL import Image
import io

def create_pdf_report(analysis_result, patient_name, sample_type, uploaded_image, grad_cam_image):
    """
    Generates a medical diagnosis report. 
    Handles conversion of Streamlit uploads to PIL images automatically.
    """
    
    # --- INTERNAL UTILITY TO ENSURE PIL IMAGE ---
    def ensure_pil(img):
        if img is None:
            return None
        # If it's a path or a Streamlit UploadedFile, open it
        if not isinstance(img, Image.Image):
            try:
                return Image.open(img).convert("RGB")
            except:
                return None
        return img.convert("RGB")

    # Pre-process images
    proc_uploaded = ensure_pil(uploaded_image)
    proc_gradcam = ensure_pil(grad_cam_image)

    class PDF(FPDF):
        def header(self):
            self.set_font('helvetica', 'B', 15)
            self.cell(0, 10, 'Pathoscope AI Diagnosis Report', 0, 1, 'C')
            self.line(10, 20, 200, 20)
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    # Initialize PDF
    pdf = PDF('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('helvetica', '', 11)
    
    # Report Metadata
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Report Generated On: {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'L')
    
    # Safely handle session state
    full_name = st.session_state.get('full_name', 'N/A')
    user_id = st.session_state.get('user_id', 'N/A')
    pdf.cell(0, 5, f"Analyst: {full_name} (ID: {user_id})", 0, 1, 'L')
    pdf.ln(5)
    
    # 1. Patient Details
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 8, '1. Patient and Sample Details', 0, 1, 'L')
    pdf.set_font('helvetica', '', 11)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(60, 7, 'Patient Name:', 1, 0, 'L', 1)
    pdf.cell(130, 7, str(patient_name), 1, 1, 'L')
    pdf.cell(60, 7, 'Sample Type:', 1, 0, 'L', 1)
    pdf.cell(130, 7, str(sample_type), 1, 1, 'L')
    pdf.cell(60, 7, 'Disease Tested:', 1, 0, 'L', 1)
    pdf.cell(130, 7, str(analysis_result.get('disease', 'N/A')), 1, 1, 'L')
    pdf.ln(5)

    # 2. Diagnosis Result
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 8, '2. Diagnosis Result', 0, 1, 'L')
    
    status_str = str(analysis_result.get('result_status', 'N/A')).upper()
    danger_keywords = ['POSITIVE', 'MALIGNANT', 'PARASITIZED', 'PNEUMONIA']
    result_color = (200, 0, 0) if any(kw in status_str for kw in danger_keywords) else (0, 120, 0)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('helvetica', '', 11)
    pdf.cell(60, 7, 'Result Status:', 1, 0, 'L', 1)
    pdf.set_text_color(*result_color)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(130, 7, status_str, 1, 1, 'L')
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('helvetica', '', 11)
    pdf.cell(60, 7, 'Confidence Score:', 1, 0, 'L', 1)
    pdf.cell(130, 7, f"{analysis_result.get('percentage', 0):.2f}%", 1, 1, 'L')
    pdf.ln(8)

    # 3. Images
    def add_img(pdf_doc, img, title):
        if img is None: return
        pdf_doc.set_font('helvetica', 'B', 11)
        pdf_doc.cell(0, 7, title, 0, 1, 'L')
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name, format="PNG")
            tmp_path = tmp.name
        
        try:
            # Automatic page break if image won't fit
            if pdf_doc.get_y() > 180:
                pdf_doc.add_page()
            pdf_doc.image(tmp_path, x=40, w=120)
            pdf_doc.ln(5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    add_img(pdf, proc_uploaded, 'A. Original Sample')
    add_img(pdf, proc_gradcam, 'B. AI Heatmap Analysis')
    
    # 4. Disclaimer
    pdf.ln(5)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, '4. Disclaimer', 0, 1, 'L')
    pdf.set_font('helvetica', 'I', 9)
    pdf.set_text_color(150, 0, 0)
    pdf.multi_cell(0, 5, "Experimental AI report. Not for final clinical diagnosis. Consult a professional.")

    # --- RETURN AS BYTES ---
    # Most reliable way to handle string vs bytes in FPDF
    try:
        raw_output = pdf.output(dest='S')
        if isinstance(raw_output, str):
            return raw_output.encode('latin-1')
        return raw_output
    except Exception as e:
        # If this fails, the error will be visible in the streamlit console
        print(f"PDF FINAL ERROR: {e}")
        raise e
        
import streamlit as st
from fpdf import FPDF
from fpdf.enums import XPos, YPos # Critical for new versions
import base64
import io

def generate_pdf_report(analysis_data, user_full_name, user_id):
    # Use 'helvetica' instead of 'Arial' to avoid font-not-found errors
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header
    pdf.set_font('helvetica', 'B', 20)
    pdf.cell(0, 15, "PATHOSCOPE AI - DIAGNOSIS REPORT", 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    pdf.set_font('helvetica', 'I', 10)
    pdf.cell(0, 10, f"Generated by: {user_full_name} (ID: {user_id})", 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')
    
    pdf.ln(10)
    
    # 2. Patient Data Table
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font('helvetica', 'B', 12)
    
    # Row 1
    pdf.cell(60, 10, "Disease Tested:", 1, 0, 'L', fill=True)
    pdf.set_font('helvetica', '', 12)
    pdf.cell(130, 10, str(analysis_data.get('disease', 'N/A')), 1, 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Row 2
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(60, 10, "Result Status:", 1, 0, 'L', fill=True)
    pdf.set_font('helvetica', '', 12)
    status = str(analysis_data.get('result_status', 'N/A')).upper()
    pdf.cell(130, 10, status, 1, 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 3. Final Summary
    pdf.ln(10)
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, "Clinical Summary:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', '', 11)
    pdf.multi_cell(0, 8, "The AI analysis indicates the presence of characteristics associated with " + 
                   str(analysis_data.get('disease', 'the selected condition')) + ". " +
                   "Please correlate these findings with clinical observations.")

    # Return the PDF as bytes
    return pdf.output()

def get_pdf_download_link(pdf_bytes, filename="Report.pdf"):
    """Generates a link allowing the PDF to be downloaded"""
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Click here to download your Report</a>'

# --- In your main UI section ---
def show_report_ui(analysis_result, user_name, uid):
    st.subheader("Generate Clinical Report")
    
    # Create the PDF in memory
    try:
        report_bytes = generate_pdf_report(analysis_result, user_name, uid)
        
        # Method 1: The standard Streamlit Download Button (Reliable)
        st.download_button(
            label="Download PDF Report",
            data=report_bytes,
            file_name=f"Pathoscope_Report_{uid}.pdf",
            mime="application/pdf"
        )
        
        # Method 2: Fallback HTML Link (If button fails)
        st.markdown(get_pdf_download_link(report_bytes), unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"PDF System Error: {str(e)}")
        st.info("The server is using a strict version of FPDF. Ensure 'new_x' and 'new_y' are used.")
        
        
                
import streamlit as st
import time
from PIL import Image

def page_login():
    """Handles user login and signup flow with DNA branding."""
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/DNA_sequence_by_Sanger_method.png/440px-DNA_sequence_by_Sanger_method.png",
        width=100
    )
    st.header("Pathoscope AI Login")

    st.info("Log in with custom credentials or use **Username: anon / Password: anon** to bypass the database check.")

    choice = st.selectbox("Action", ["Login", "Sign Up"], key="auth_choice")

    if choice == "Login":
        with st.form("login_form"):
            st.markdown("#### User Login")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                with st.spinner("Authenticating..."):
                    # This calls the auth function from your logic block
                    user_id, full_name = authenticate_user(username, password)
                    if user_id:
                        st.session_state['logged_in'] = True
                        st.session_state['user_id'] = user_id
                        st.session_state['full_name'] = full_name
                        st.success(f"Welcome back, {full_name}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password.")

    elif choice == "Sign Up":
        with st.form("signup_form"):
            st.markdown("#### New User Registration")
            new_full_name = st.text_input("Full Name", key="signup_name")
            new_username = st.text_input("New Username", key="signup_user")
            new_password = st.text_input("New Password", type="password", key="signup_pass")
            submitted = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submitted:
                if new_username and new_password and new_full_name:
                    if len(new_password) < 6:
                        st.warning("Password must be at least 6 characters.")
                    else:
                        with st.spinner("Creating account..."):
                            save_user(new_username, new_password, new_full_name)
                            st.success("Account created! Please switch to Login.")
                else:
                    st.warning("Please fill in all fields.")

def page_diagnosis():
    """Main Diagnosis page for image upload, AI analysis, and report generation."""
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/DNA_sequence_by_Sanger_method.png/440px-DNA_sequence_by_Sanger_method.png",
        width=100
    )
    st.title("Smart Microscope Diagnosis System")
    st.markdown("Analyze microscope images to detect diseases and visualize infected regions using deep learning.")

    # Show model architecture details
    if 'display_training_structure' in globals():
        display_training_structure()
    
    # Section: Patient Details
    st.subheader("Patient & Sample Details")
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name:", key="patient_name_input", value=st.session_state.get('full_name', ''))
    with col2:
        sample_type = st.selectbox(
            "Sample Type",
            get_sample_options() if 'get_sample_options' in globals() else ["Blood Smear", "Tissue Biopsy", "Cell Culture"],
            key="sample_type_select"
        )
    
    # Determine the disease category
    disease_tested = get_disease_from_sample(sample_type) if 'get_disease_from_sample' in globals() else "Unknown"

    # Load AI Model (Internal logic handles real vs mock)
    model = load_model_real(disease_tested) if 'load_model_real' in globals() else None

    # Section: Image Input
    st.subheader("Choose Image Source")
    input_method = st.radio(
        "Select Input Method:",
        ("Upload Image from Device", "Capture Live via Microscope/Camera"),
        key="input_method_radio",
        horizontal=True
    )
    
    uploaded_file = None
    if input_method == "Upload Image from Device":
        uploaded_file = st.file_uploader("Upload microscope image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    else:
        uploaded_file = st.camera_input("Take Picture (Microscope View)")
        
    st.markdown("---")
    
    # Analysis Trigger
    analyze_button = st.button("🔬 Start AI Analysis", use_container_width=True, disabled=(uploaded_file is None or not patient_name))
    
    if analyze_button and uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Current Sample View", use_container_width=True)
            
            # Cache inputs for report
            st.session_state['uploaded_image'] = image
            st.session_state['patient_name_report'] = patient_name
            st.session_state['sample_type_report'] = sample_type
            
            with st.spinner(f"Running Inference for {disease_tested}..."):
                # Perform prediction
                analysis_result = get_real_prediction(model, image, disease_tested)
                st.session_state['analysis_result'] = analysis_result
                
                # UI Results Display
                st.header("Diagnosis Result")
                status = analysis_result['result_status']
                confidence = analysis_result['percentage']
                
                # Check for positive/negative keywords to colorize
                danger_words = ['POSITIVE', 'MALIGNANT', 'PARASITIZED']
                if any(w in status.upper() for w in danger_words):
                    st.error(f"Status: **{status}** (Confidence: {confidence:.2f}%)")
                else:
                    st.success(f"Status: **{status}** (Confidence: {confidence:.2f}%)")

                # Show Explainability Map
                st.subheader("Explainability: Grad-CAM Visualization")
                grad_cam_img = analysis_result['grad_cam_image']
                st.image(grad_cam_img, caption="Activation Map (Affected Areas in Red/Yellow)", use_container_width=True)
                st.session_state['grad_cam_image'] = grad_cam_img
                
                # Persist to history
                save_diagnosis(
                    st.session_state['user_id'],
                    patient_name,
                    sample_type,
                    disease_tested,
                    status,
                    analysis_result['confidence_decimal'],
                    image_path="internal_storage"
                )
        except Exception as e:
            st.error(f"Processing Error: {e}")

    # Section: Report Generation
    if 'analysis_result' in st.session_state:
        st.markdown("---")
        st.subheader("Generate Clinical Report")
        
        try:
            # Call our PDF generation tool
            pdf_data = create_pdf_report(
                st.session_state['analysis_result'],
                st.session_state.get('patient_name_report', 'N/A'),
                st.session_state.get('sample_type_report', 'N/A'),
                st.session_state.get('uploaded_image'),
                st.session_state.get('grad_cam_image')
            )
            
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_data,
                file_name=f"Pathoscope_{patient_name}_{int(time.time())}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.warning("PDF Generation is currently unavailable in this environment.")

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time

def page_history():
    """History page to view past diagnosis records with Voice Search."""
    st.title("📚 Diagnosis History")
    st.info(f"Viewing history for: **{st.session_state.get('full_name', 'User')}**")

    # Guard against non-persistent sessions
    if st.session_state.get('user_id') == 9999:
        st.warning("History is not saved for anonymous users. Please log in with a registered account.")
        return

    # Fetch records from the database logic built in previous blocks
    records = fetch_history(st.session_state['user_id']) if 'fetch_history' in globals() else []

    if not records:
        st.warning("No diagnosis history found for this user. Records appear here after an analysis is completed.")
        return

    # --- VOICE SEARCH UI ---
    st.subheader("Search Records")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # The key "voice_search_input" is used by the JS to inject text
        search_query = st.text_input("Enter patient name or disease", key="voice_search_input", placeholder="Type or use microphone...")
    
    with col2:
        st.write(" ") # Visual spacer
        # Speech-to-Text Bridge: Injects transcribed text directly into the Streamlit input DOM
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
                    
                    // Logic to find the Streamlit input field and simulate user input
                    const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                    let targetInput = null;

                    for (let input of inputs) {
                        if (input.placeholder === "Type or use microphone...") {
                            targetInput = input;
                            break;
                        }
                    }

                    if (targetInput) {
                        const nativeValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                        nativeValueSetter.call(targetInput, voiceText);

                        // Trigger React/Streamlit change detection
                        const event = new Event('input', { bubbles: true });
                        targetInput.dispatchEvent(event);
                        
                        targetInput.focus();
                        setTimeout(() => {
                            targetInput.blur();
                        }, 100);
                    }
                };
                
                recognition.onerror = function(e) {
                    console.error("Speech Error: ", e.error);
                    recognition.stop();
                };
            } else {
                alert("Voice features require Chrome or Edge.");
            }
        }
        </script>
        <button onclick="startDictation()" style="background-color: #ff4b4b; color: white; border: none; padding: 10px; border-radius: 8px; cursor: pointer; width: 100%; height: 45px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">
            🎤 Speak
        </button>
        """
        components.html(voice_js, height=45)

    # --- DATA PROCESSING & FILTERING ---
    history_data = []
    for record in records:
        history_data.append({
            "Date": record['diagnosis_date'].strftime('%Y-%m-%d %H:%M') if hasattr(record['diagnosis_date'], 'strftime') else str(record['diagnosis_date']),
            "Patient": record['patient_name'],
            "Sample": record['sample_type'],
            "Disease": record['disease_tested'],
            "Result": record['result_status'],
            "Confidence": f"{record['confidence_score']*100:.1f}%",
        })
    
    df = pd.DataFrame(history_data)

    # Filter based on search bar (Real-time sync)
    if search_query:
        df = df[
            df['Patient'].str.contains(search_query, case=False) | 
            df['Disease'].str.contains(search_query, case=False) |
            df['Result'].str.contains(search_query, case=False)
        ]

    st.subheader(f"Results Found: {len(df)}")
    st.dataframe(
        df, 
        use_container_width=True,
        hide_index=True 
    )

def main():
    """Main Application Entry Point."""
    st.set_page_config(
        page_title="Pathoscope AI",
        page_icon="🔬",
        layout="wide"
    )
    
    # Global state init
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # Sidebar Navigation
    st.sidebar.title("🔬 Pathoscope AI")
    st.sidebar.caption("v1.2.0 | Next-Gen Diagnostics")
    st.sidebar.markdown("---")

    if not st.session_state['logged_in']:
        # If not logged in, show the login page (Block 5)
        page_login()
    else:
        # User is authenticated
        st.sidebar.success(f"Logged in: {st.session_state['full_name']}")
        
        app_page = st.sidebar.radio(
            "Navigate",
            ["Diagnosis", "History"],
            key="main_nav_radio"
        )
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🔒 Secure Logout", use_container_width=True):
            # Clear critical session keys
            for key in ['logged_in', 'user_id', 'full_name', 'analysis_result']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['logged_in'] = False
            st.rerun()

        # Page Router
        if app_page == "Diagnosis":
            page_diagnosis() # From Block 5
        elif app_page == "History":
            page_history()

if __name__ == '__main__':
    main()







