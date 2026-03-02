        import streamlit as st
from PIL import Image
import time
import hashlib 
import mysql.connector
from ml_logic import (
    get_sample_options, 
    display_training_structure, 
    load_model_real, # PyTorch function to load the ML model
    get_real_prediction, # PyTorch function for real (or simulated) inference
    get_disease_from_sample,
    DISEASE_MODELS # Used for configuration details
)
import streamlit.components.v1 as components
import pandas as pd 
import io
# import base64 # Removed, as we are no longer using Base64 URI embedding
from fpdf import FPDF 
import tempfile # IMPORTANT: Import for creating temporary files
import os # IMPORTANT: Import for removing temporary files

# --- CONFIGURATION (MUST CHANGE) ---
# IMPORTANT: Update these values with your MySQL setup. 
# WARNING: Using 'root' and simple passwords is NOT secure for production.
MYSQL_CONFIG = {
    'user': 'root', 
    'password': 'Kiruthika0104!', 
    'host': 'localhost',
    'database': 'pathoscope_db' # Ensure this database exists
}
# -----------------------------------

# --- Database Connection and Utility Functions ---

def get_db_connection():
    """Establishes a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except mysql.connector.Error as err:
        # Display a warning if the DB connection fails
        st.warning(f"Database Connection Warning: {err}. Proceeding without persistence features.")
        return None

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
        st.warning("Cannot connect to database. Signup skipped.")
        return False
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            st.warning("Username already exists. Please choose a different one.")
            return False
        password_hash = hash_password(password)
        # Note: The 'users' table must be created beforehand in your MySQL database.
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
    if not conn: 
        # Fallback to anonymous use if DB is down.
        if username == "anon" and password == "anon":
            return 9999, "Anonymous User"
        return None, None

    cursor = conn.cursor(dictionary=True)
    try:
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
        cursor.close()
        conn.close()

def save_diagnosis(user_id, patient_name, sample_type, disease_tested, result_status, confidence_score, image_path="N/A"):
    """Saves a diagnosis record to the database."""
    conn = get_db_connection()
    if not conn: return True # Return True to allow UI to proceed even without DB
    cursor = conn.cursor()
    try:
        # Note: The 'patient_history' table must be created beforehand.
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
        cursor.close()
        conn.close()

def fetch_history(user_id):
    """Fetches all diagnosis records for the logged-in user."""
    conn = get_db_connection()
    if not conn: return []
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
        SELECT * FROM patient_history 
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
        cursor.close()
        conn.close()

def create_pdf_report(analysis_result, patient_name, sample_type, uploaded_image, grad_cam_image):
    """Generates a detailed diagnosis report in PDF format using FPDF."""
    
    class PDF(FPDF):
        def header(self):
            # Logo (Simulated)
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
    pdf.set_text_color(100, 100, 100) # Gray
    pdf.cell(0, 5, f"Report Generated On: {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'L')
    pdf.cell(0, 5, f"Analyst: {st.session_state.get('full_name', 'N/A')} (ID: {st.session_state.get('user_id', 'N/A')})", 0, 1, 'L')
    pdf.ln(5)
    
    # 1. Patient and Sample Details
    pdf.set_text_color(0, 0, 0) # Black
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, '1. Patient and Sample Details', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    # Detail Table (simple grid)
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

    result_color = (255, 0, 0) if 'POSITIVE' in analysis_result['result_status'].upper() or 'MALIGNANT' in analysis_result['result_status'].upper() or 'PARASITIZED' in analysis_result['result_status'].upper() else (0, 128, 0)
    
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
        """Helper function to save image to a temp file and embed it in the PDF."""
        pdf_doc.set_font('Arial', 'BU', 11)
        pdf_doc.cell(0, 7, title, 0, 1, 'L')
        pdf_doc.set_font('Arial', 'I', 9)
        pdf_doc.multi_cell(0, 5, caption, 0, 'L')
        pdf_doc.ln(2)
        
        if image_obj:
            temp_file_path = None
            try:
                # 1. Create a temporary file with a .png extension
                # delete=False means the file is created on the filesystem and we need to clean it up
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_file_path = tmp.name
                    # 2. Save the PIL Image object directly to the temporary file path
                    image_obj.save(temp_file_path, format="PNG") 
                
                # 3. Use pdf.image() with the actual file path
                # Use the path in pdf.image(). w=90 is the width in mm.
                pdf_doc.image(temp_file_path, w=90) 
                
            except Exception as e:
                pdf_doc.set_font('Arial', 'I', 10)
                pdf_doc.cell(0, 5, f'Error embedding image: {e}', 0, 1, 'L')
                
            finally:
                # 4. CRITICAL: Clean up the temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
            pdf_doc.ln(5)
        else:
            pdf_doc.set_font('Arial', 'I', 10)
            pdf_doc.cell(0, 5, 'Image not available.', 0, 1, 'L')
            pdf_doc.ln(5)
            
    # Add Original Image
    add_image_to_pdf(
        pdf, 
        uploaded_image, 
        'A. Original Sample Image', 
        'The raw microscopic image uploaded for analysis.'
    )
    
    # Add Grad-CAM Image
    add_image_to_pdf(
        pdf, 
        grad_cam_image, 
        'B. Grad-CAM Visualization', 
        'Regions (red/yellow) indicate where the AI model focused its attention for the prediction.'
    )
    
    # 4. Disclaimer
    # Check if a new page is needed, otherwise place the disclaimer on the current page
    if pdf.get_y() > 250:
        pdf.add_page()
    else:
        pdf.set_y(pdf.get_y() + 10)
        
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, '4. Disclaimer', 0, 1, 'L')
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(150, 0, 0) # Reddish
    disclaimer = ("This report is for informational purposes only and is based on an automated AI analysis model. "
                  "It is NOT a substitute for professional medical diagnosis or consultation. "
                  "All results must be verified and confirmed by a qualified pathologist.")
    pdf.multi_cell(0, 6, disclaimer, 0, 'J')
    
    # Output the PDF as bytes
    return pdf.output(dest='S').encode('latin1')

# --- Streamlit Page Functions (rest of the code remains the same) ---

def page_login():
    """Handles user login and signup flow."""
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
                    user_id, full_name = authenticate_user(username, password)
                    if user_id:
                        st.session_state['logged_in'] = True
                        st.session_state['user_id'] = user_id
                        st.session_state['full_name'] = full_name
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
                else:
                    st.warning("Please fill in all fields.")

def page_diagnosis():
    """Diagnosis page for image upload and analysis."""
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/DNA_sequence_by_Sanger_method.png/440px-DNA_sequence_by_Sanger_method.png",
        width=100
    )
    st.title("Smart Microscope Diagnosis System")
    st.markdown("Analyze microscope images to detect diseases and visualize infected regions.")

    # Display the ML training structure
    display_training_structure()
    
    # Patient Details
    st.subheader("Patient & Sample Details")
    col1, col2 = st.columns(2)
    with col1:
        # Pre-fill patient name with user's full name if available
        patient_name = st.text_input("Patient Name:", key="patient_name_input", value=st.session_state.get('full_name', ''))
    with col2:
        # Use the imported function to get sample options
        sample_type = st.selectbox(
            "Sample Type",
            get_sample_options(),
            key="sample_type_select"
        )
    
    # Determine the disease based on sample type
    disease_tested = get_disease_from_sample(sample_type)

    # 1. Load Model for the selected disease 
    # This will load the real model if the file exists and is valid
    model = load_model_real(disease_tested) 

    # Image Input
    st.subheader("Choose Image Source")
    
    input_method = st.radio(
        "Select Input Method:",
        ("Upload Image from Device", "Capture Live via Microscope/Camera"),
        key="input_method_radio"
    )
    
    uploaded_file = None
    
    if input_method == "Upload Image from Device":
        uploaded_file = st.file_uploader(
            "Upload microscope image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"]
        )
    elif input_method == "Capture Live via Microscope/Camera":
        uploaded_file = st.camera_input("Take Picture (Use this for 'Live' Sample)")
        
    st.markdown("---")
    
    # The analyze button should only be active if an image is present and patient name is entered
    analyze_button = st.button("🔬 Analyze Sample", use_container_width=True, disabled=(uploaded_file is None or not patient_name))
    
    if analyze_button and uploaded_file is not None and patient_name:
        
        # 1. Process Uploaded Image
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Save original image to session state for report generation
            st.session_state['uploaded_image'] = image
            st.session_state['patient_name_report'] = patient_name
            st.session_state['sample_type_report'] = sample_type
            
        except Exception as e:
            st.error(f"Error reading image: {e}")
            return
            
        st.info(f"Running **{disease_tested}** Model Inference...")
        
        # 3. Run REAL Prediction (or fallback to Deterministic Mock)
        with st.spinner(f"Analyzing Image using {DISEASE_MODELS[disease_tested]['model_file']}..."):
            # If model is None (failed load), get_real_prediction handles the simulation fallback
            analysis_result = get_real_prediction(model, image, disease_tested)
        
        # Save analysis result to session state
        st.session_state['analysis_result'] = analysis_result
        
        # 4. Display Results
        st.header("Diagnosis Result")
        
        result_status = analysis_result['result_status']
        percentage = analysis_result['percentage']
        
        # Determine the positive class name for the specific disease tested
        config = DISEASE_MODELS.get(disease_tested)
        is_positive_indicator = False

        if config and len(config['target_classes']) == 2:
            # The 'positive' result is always the second element (index 1) in target_classes 
            positive_class = config['target_classes'][1]
            
            # Check if the result matches the defined positive class
            if result_status == positive_class:
                is_positive_indicator = True

        if is_positive_indicator:
            st.error(f"Result: **POSITIVE INDICATOR** for {disease_tested} ({result_status}) (Confidence: {percentage:.2f}%)")
        else:
            # This covers the negative class (index 0)
            st.success(f"Result: **NEGATIVE INDICATOR** for {disease_tested} ({result_status}) (Confidence: {percentage:.2f}%)")
            
        # 5. Display Grad-CAM (Processed Image)
        st.subheader("Grad-CAM Visualization (Affected Regions)")
        st.info("The Grad-CAM overlay (red/yellow) indicates the high-activation regions identified by the model.")
        
        grad_cam_image = analysis_result['grad_cam_image']
        st.image(grad_cam_image, caption="Processed Image with Simulated Grad-CAM Overlay", use_container_width=True)
        
        # Save grad-cam image to session state for report generation
        st.session_state['grad_cam_image'] = grad_cam_image
        
        # 6. Save History
        if save_diagnosis(
            st.session_state['user_id'],
            patient_name,
            sample_type,
            analysis_result['disease'],
            analysis_result['result_status'],
            analysis_result['confidence_decimal'],
            # Using a simple unique identifier for the image path since we don't save the file itself
            image_path=f"{sample_type}_{result_status}_{time.time()}" 
        ):
            st.success("Diagnosis record saved to your history.")

    # --- Report Download Section ---
    if 'analysis_result' in st.session_state and st.session_state['analysis_result']:
        
        uploaded_img = st.session_state.get('uploaded_image')
        grad_cam_img = st.session_state.get('grad_cam_image')
        patient_name_report = st.session_state.get('patient_name_report', 'Unknown_Patient')
        sample_type_report = st.session_state.get('sample_type_report', 'Unknown_Sample')

        # Generate the PDF report content
        pdf_bytes = None
        try:
            pdf_bytes = create_pdf_report(
                st.session_state['analysis_result'], 
                patient_name_report, 
                sample_type_report, 
                uploaded_img, 
                grad_cam_img
            )
        except Exception as e:
            st.error(f"Error generating PDF report. Please check server permissions for temp files. Error: {e}")
            
        if pdf_bytes:
            st.markdown("---")
            st.subheader("Download Report")

            # Use the download button to provide the PDF file
            st.download_button(
                label="⬇️ Download Detailed Report (PDF)",
                data=pdf_bytes,
                file_name=f"Pathoscope_Report_{patient_name_report}_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            st.info("The downloaded file is a structured PDF report containing all patient details, results, and images.")


import streamlit.components.v1 as components

# ... existing imports ...

def page_history():
    """History page to view past diagnosis records with Voice Search."""
    st.title("📚 Diagnosis History")
    st.info(f"Viewing history for: **{st.session_state['full_name']}**")

    if st.session_state['user_id'] == 9999:
        st.warning("History is not saved for anonymous users. Please log in with a registered account.")
        return

    records = fetch_history(st.session_state['user_id'])

    if not records:
        st.warning("No diagnosis history found for this user.")
        return

    # --- NEW: SEARCH BAR AND VOICE COMPONENT ---
    st.subheader("Search Records")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # The key "voice_search_input" is used by the JS to inject text
        search_query = st.text_input("Enter patient name or disease", key="voice_search_input")
    
    with col2:
        st.write(" ") # Alignment
        # Voice Recognition JavaScript
        # Updated Voice Recognition JavaScript with React-specific state forcing
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
                    
                    // Search for the input field in the parent document (Streamlit runs in an iframe)
                    const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                    let targetInput = null;

                    for (let input of inputs) {
                        // Find the input by label or placeholder
                        if (input.getAttribute('aria-label') === "Enter patient name or disease" || 
                            input.placeholder === "Enter patient name or disease") {
                            targetInput = input;
                            break;
                        }
                    }

                    if (targetInput) {
                        // 1. Force the value into the input
                        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                        nativeInputValueSetter.call(targetInput, voiceText);

                        // 2. Dispatch 'input' event to notify React/Streamlit
                        const event = new Event('input', { bubbles: true });
                        targetInput.dispatchEvent(event);

                        // 3. Focus and Blur to trigger Streamlit's 'onChange' sync
                        targetInput.focus();
                        
                        // Small timeout to ensure the focus is registered before blurring
                        setTimeout(() => {
                            targetInput.blur();
                            
                            // 4. Send an Enter key press as a final fallback
                            targetInput.dispatchEvent(new KeyboardEvent('keydown', {
                                key: 'Enter',
                                code: 'Enter',
                                keyCode: 13,
                                which: 13,
                                bubbles: true
                            }));
                        }, 100);
                    }
                };
                
                recognition.onerror = function(e) {
                    console.error("Speech Recognition Error: ", e.error);
                    recognition.stop();
                };
            } else {
                alert("Speech recognition not supported in this browser. Please use Chrome or Edge.");
            }
        }
        </script>
        <button onclick="startDictation()" style="background-color: #ff4b4b; color: white; border: none; padding: 10px; border-radius: 8px; cursor: pointer; width: 100%; height: 45px; display: flex; align-items: center; justify-content: center; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            🎤 Click to Speak 
        </button>
        """
        components.html(voice_js, height=45)

    # Convert to DataFrame
    history_data = []
    for record in records:
        history_data.append({
            "Date": record['diagnosis_date'].strftime('%Y-%m-%d %H:%M:%S'),
            "Patient": record['patient_name'],
            "Sample": record['sample_type'],
            "Disease": record['disease_tested'],
            "Result": record['result_status'],
            "Confidence (%)": f"{record['confidence_score']*100:.2f}",
        })
    
    df = pd.DataFrame(history_data)

    # --- NEW: FILTERING LOGIC ---
    if search_query:
        df = df[
            df['Patient'].str.contains(search_query, case=False) | 
            df['Disease'].str.contains(search_query, case=False) |
            df['Result'].str.contains(search_query, case=False)
        ]

    st.subheader(f"Total Records: {len(df)}")
    st.dataframe(
        df, 
        use_container_width=True,
        hide_index=True 
    )

# --- Main Application Logic ---

def main():
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Pathoscope AI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for login/navigation
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['page'] = 'Login'

    st.sidebar.title("Pathoscope AI")
    st.sidebar.markdown("---")

    if not st.session_state['logged_in']:
        page_login()
    else:
        st.sidebar.markdown(f"**Welcome, {st.session_state['full_name']}**")
        st.sidebar.markdown("---")
        
        app_page = st.sidebar.radio(
            "Navigate",
            ["Diagnosis", "History"],
            key="main_nav_radio"
        )
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🔒 Logout", use_container_width=True):
            # Clear all session state for a clean logout
            keys_to_delete = [k for k in st.session_state if k not in ['logged_in', 'page']]
            for k in keys_to_delete:
                del st.session_state[k]
            st.session_state['logged_in'] = False
            st.rerun()

        if app_page == "Diagnosis":
            page_diagnosis()
        elif app_page == "History":
            page_history()
            

if __name__ == '__main__':
    main()
