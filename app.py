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
import tempfile 
import os 
from datetime import datetime

# Switched to ReportLab for cloud stability (replaces fpdf)
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- UPDATED CONFIGURATION (AIVEN CLOUD READY) ---
def get_db_connection():
    """Establishes a connection to the Aiven MySQL database using secrets."""
    try:
        conn = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            port=int(st.secrets["mysql"]["port"]),
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        return conn
    except Exception as e:
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
    
    if not conn: 
        if username == "admin" and password == "admin123":
            st.info("Emergency Login: Using offline admin credentials.")
            return 9999, "Offline Administrator"
        return None, None

    try:
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
        if conn:
            conn.close()

def save_diagnosis(user_id, patient_name, sample_type, disease_tested, result_status, confidence_score, image_path="N/A"):
    """Saves a diagnosis record to the database."""
    conn = get_db_connection()
    if not conn:
        return True 
    
    try:
        cursor = conn.cursor()
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
        if conn:
            conn.close()

def fetch_history(user_id):
    """Fetches all diagnosis records for the logged-in user."""
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
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
        if conn:
            conn.close()

def create_pdf_report(analysis_result, patient_name, sample_type, uploaded_image, grad_cam_image):
    """
    Generates a medical diagnosis report using ReportLab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, spaceAfter=12)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], spaceBefore=10, spaceAfter=10)
    normal_style = styles['Normal']
    warning_style = ParagraphStyle('WarningStyle', parent=styles['Italic'], textColor=colors.red, fontSize=9)

    elements = []

    # Header
    elements.append(Paragraph("Pathoscope AI Diagnosis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Metadata
    full_name = st.session_state.get('full_name', 'N/A')
    user_id = st.session_state.get('user_id', 'N/A')
    elements.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Paragraph(f"Analyst: {full_name} (ID: {user_id})", normal_style))
    elements.append(Spacer(1, 12))

    # 1. Patient Details Table
    elements.append(Paragraph("1. Patient and Sample Details", heading_style))
    data = [
        ["Patient Name:", str(patient_name)],
        ["Sample Type:", str(sample_type)],
        ["Disease Tested:", str(analysis_result.get('disease', 'N/A'))]
    ]
    t = Table(data, colWidths=[150, 300])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    # 2. Diagnosis Result
    elements.append(Paragraph("2. Diagnosis Result", heading_style))
    status_str = str(analysis_result.get('result_status', 'N/A')).upper()
    danger_keywords = ['POSITIVE', 'MALIGNANT', 'PARASITIZED', 'PNEUMONIA']
    res_color = colors.red if any(kw in status_str for kw in danger_keywords) else colors.green
    
    res_data = [
        ["Result Status:", status_str],
        ["Confidence Score:", f"{analysis_result.get('percentage', 0):.2f}%"]
    ]
    rt = Table(res_data, colWidths=[150, 300])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
        ('TEXTCOLOR', (1, 0), (1, 0), res_color),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(rt)
    elements.append(Spacer(1, 12))

    # 3. Images
    elements.append(Paragraph("3. Visual Analysis", heading_style))
    
    from reportlab.platypus import Image as RLImage
    
    def process_img_for_pdf(img):
        if img is None: return None
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name, format="PNG")
        return tmp.name

    orig_path = process_img_for_pdf(uploaded_image)
    grad_path = process_img_for_pdf(grad_cam_image)

    if orig_path:
        elements.append(Paragraph("A. Original Sample", styles['Normal']))
        elements.append(RLImage(orig_path, width=300, height=200))
        elements.append(Spacer(1, 12))
    
    if grad_path:
        elements.append(Paragraph("B. AI Heatmap Analysis", styles['Normal']))
        elements.append(RLImage(grad_path, width=300, height=200))
        elements.append(Spacer(1, 12))

    # 4. Disclaimer
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("4. Disclaimer", heading_style))
    elements.append(Paragraph("Experimental AI report. Not for final clinical diagnosis. Consult a professional medical practitioner.", warning_style))

    # Build PDF
    doc.build(elements)
    
    # Cleanup temp files
    for p in [orig_path, grad_path]:
        if p and os.path.exists(p): os.remove(p)
        
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
        
import streamlit as st
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime

def generate_reportlab_pdf(analysis_data, user_full_name, user_id):
    """
    Generates a professional medical report using ReportLab.
    Stays entirely in memory using BytesIO.
    """
    buffer = io.BytesIO()
    
    # Create the document template
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        rightMargin=50, 
        leftMargin=50, 
        topMargin=50, 
        bottomMargin=50
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    # --- 1. Custom Styles ---
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor("#2E5077"),
        alignment=1, # Center
        spaceAfter=20
    )
    
    header_style = ParagraphStyle(
        'HeaderInfo',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=2 # Right
    )

    # --- 2. Header Content ---
    elements.append(Paragraph("PATHOSCOPE AI - DIAGNOSIS REPORT", title_style))
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_info = f"Date: {timestamp}<br/>Analyst: {user_full_name} (ID: {user_id})"
    elements.append(Paragraph(meta_info, header_style))
    elements.append(Spacer(1, 20))
    
    # --- 3. Patient & Result Table ---
    # Determine color based on result status
    status_str = str(analysis_data.get('result_status', '')).upper()
    danger_keywords = ['POSITIVE', 'MALIGNANT', 'PARASITIZED', 'PNEUMONIA']
    is_danger = any(kw in status_str for kw in danger_keywords)
    result_text_color = colors.red if is_danger else colors.green

    table_data = [
        [Paragraph("<b>PARAMETER</b>", styles['Normal']), Paragraph("<b>CLINICAL VALUE</b>", styles['Normal'])],
        ["Patient Name", str(analysis_data.get('patient_name', 'N/A'))],
        ["Sample Type", str(analysis_data.get('sample_type', 'N/A'))],
        ["Disease Tested", str(analysis_data.get('disease', 'N/A'))],
        ["Result Status", status_str],
        ["Confidence Score", f"{analysis_data.get('percentage', 0):.2f}%"]
    ]
    
    # Style the table
    t = Table(table_data, colWidths=[150, 300])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        # Set the color for the "Result Status" value cell specifically
        ('TEXTCOLOR', (1, 4), (1, 4), result_text_color),
        ('FONTNAME', (1, 4), (1, 4), 'Helvetica-Bold'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 30))
    
    # --- 4. Clinical Summary ---
    elements.append(Paragraph("<b>4. Clinical Summary</b>", styles['Heading2']))
    disease_name = analysis_data.get('disease', 'the target condition')
    summary_text = (
        f"The AI analysis has identified markers consistent with "
        f"<b>{disease_name}</b>. "
        "These findings are based on digital image processing and should be correlated "
        "with patient history, clinical symptoms, and secondary diagnostic tests."
    )
    elements.append(Paragraph(summary_text, styles['Normal']))
    
    elements.append(Spacer(1, 40))
    
    # --- 5. Disclaimer ---
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Italic'], fontSize=8, textColor=colors.red)
    disclaimer_text = (
        "DISCLAIMER: This report is generated by an artificial intelligence system. "
        "It is intended for clinical assistance only and does not constitute a final "
        "medical diagnosis. Please consult with a board-certified pathologist."
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build the PDF
    doc.build(elements)
    
    # Get bytes from buffer
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def get_reportlab_download_link(pdf_bytes, filename="Pathoscope_Report.pdf"):
    """Creates a base64 encoded download link for the PDF."""
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'''
        <a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration:none;">
            <button style="background-color:#007BFF; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer; font-weight:bold;">
                Download PDF Report (Link)
            </button>
        </a>
    '''

def show_clinical_report_ui(results, name, uid):
    """Main UI component to generate and download the report."""
    st.write("---")
    st.subheader("📋 Digital Lab Report")
    
    with st.spinner("Generating clinical report..."):
        try:
            # Ensure patient name is included in results for the PDF function
            results['patient_name'] = name
            
            pdf_content = generate_reportlab_pdf(results, st.session_state.get('full_name', 'Analyst'), uid)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("Your report is ready for download.")
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_content,
                    file_name=f"Report_{name.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    key="reportlab_download_btn"
                )
            
            with col2:
                st.write("Browser Fallback:")
                st.markdown(get_reportlab_download_link(pdf_content), unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Failed to generate Report: {str(e)}")
        
                
import streamlit as st
import time
from PIL import Image

def page_login():
    """Handles user login and signup flow with DNA branding."""
    st.markdown("<br>", unsafe_allow_html=True)
    col_logo, col_text = st.columns([1, 4])
    with col_logo:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/DNA_sequence_by_Sanger_method.png/440px-DNA_sequence_by_Sanger_method.png",
            width=80
        )
    with col_text:
        st.title("Pathoscope AI")
        st.caption("Next-Generation Digital Pathology & Diagnostic Assistant")

    st.info("Log in with custom credentials or use **Username: anon / Password: anon** to bypass the database check.")

    choice = st.radio("Access Portal", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")

    if choice == "Login":
        with st.form("login_form"):
            st.markdown("### 🔐 User Login")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Authenticate", use_container_width=True)
            
            if submitted:
                if not username or not password:
                    st.warning("Please enter both username and password.")
                else:
                    with st.spinner("Authenticating..."):
                        # References Block 2's authenticate_user
                        user_id, full_name = authenticate_user(username, password)
                        if user_id:
                            st.session_state['logged_in'] = True
                            st.session_state['user_id'] = user_id
                            st.session_state['full_name'] = full_name
                            st.success(f"Welcome back, {full_name}!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Invalid Username or Password.")

    elif choice == "Sign Up":
        with st.form("signup_form"):
            st.markdown("### 📝 New Registration")
            new_full_name = st.text_input("Full Name (e.g., Dr. John Doe)", key="signup_name")
            new_username = st.text_input("Choose Username", key="signup_user")
            new_password = st.text_input("Choose Password", type="password", key="signup_pass")
            submitted = st.form_submit_button("Register Account", use_container_width=True)
            
            if submitted:
                if new_username and new_password and new_full_name:
                    if len(new_password) < 4:
                        st.warning("Password is too short.")
                    else:
                        with st.spinner("Creating account..."):
                            # References Block 2's save_user
                            save_user(new_username, new_password, new_full_name)
                            st.success("Account created! You can now switch to Login.")
                else:
                    st.warning("Please fill in all fields.")

def page_diagnosis():
    """Main Diagnosis page for image upload, AI analysis, and report generation."""
    st.title("🔬 Smart Microscope Diagnosis")
    st.markdown("Upload or capture microscope imagery to detect anomalies using Deep Learning.")

    # 1. Patient Details
    with st.expander("👤 Patient & Sample Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Full Name:", key="patient_name_input")
        with col2:
            # Fallback for sample options if Block 1 helper isn't available
            sample_options = ["Blood Smear (Malaria)", "Tissue Biopsy (Cancer)", "Chest X-Ray (Pneumonia)"]
            sample_type = st.selectbox("Sample Source", sample_options, key="sample_type_select")
    
    # Logic from Block 1/2 to determine disease
    disease_tested = sample_type.split('(')[-1].strip(')') if '(' in sample_type else "General"

    # 2. Image Input
    st.subheader("📸 Image Acquisition")
    input_method = st.tabs(["📁 File Upload", "📷 Live Camera"])
    
    uploaded_file = None
    with input_method[0]:
        uploaded_file = st.file_uploader("Select Microscope Image", type=["png", "jpg", "jpeg"])
    with input_method[1]:
        camera_file = st.camera_input("Microscope Camera Feed")
        if camera_file:
            uploaded_file = camera_file
        
    # 3. Execution
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Current View", use_container_width=True)
        
        analyze_button = st.button("🚀 Analyze Sample", use_container_width=True, type="primary")
        
        if analyze_button:
            if not patient_name:
                st.warning("Please enter a patient name before analyzing.")
            else:
                with st.spinner(f"Processing {disease_tested} analysis..."):
                    # References Block 2's get_prediction logic
                    # We pass the sample type to handle mock vs real data correctly
                    results = get_prediction(image, sample_type)
                    
                    st.divider()
                    st.header("Results Summary")
                    
                    res_col1, res_col2 = st.columns(2)
                    status = results['result_status']
                    conf = results['percentage']
                    
                    with res_col1:
                        # Visual indicators for results
                        is_danger = any(kw in status.upper() for kw in ['POSITIVE', 'MALIGNANT', 'PNEUMONIA'])
                        if is_danger:
                            st.error(f"**FINDING:** {status}")
                        else:
                            st.success(f"**FINDING:** {status}")
                        st.metric("Confidence Score", f"{conf:.2f}%")
                    
                    with res_col2:
                        st.write("**Explainability Heatmap (Grad-CAM)**")
                        # Show the Grad-CAM image returned by Block 2
                        st.image(results['grad_cam_image'], use_container_width=True)

                    # 4. Report Generation (Block 3 Integration)
                    # We pass the data to the UI helper from Block 3
                    if 'show_clinical_report_ui' in globals():
                        show_clinical_report_ui(results, patient_name, st.session_state.get('user_id', 'UID-000'))
                    
                    # 5. Save to History (Block 2 Integration)
                    if 'save_diagnosis' in globals():
                        save_diagnosis(
                            st.session_state['user_id'],
                            patient_name,
                            sample_type,
                            disease_tested,
                            status,
                            results['confidence_decimal']
                        )
    else:
        st.info("Waiting for image input to begin analysis.")


import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time

def page_history():
    """History page to view past diagnosis records with Voice Search."""
    st.title("📚 Diagnosis History")
    
    user_name = st.session_state.get('full_name', 'User')
    user_id = st.session_state.get('user_id')
    
    st.info(f"Viewing clinical history for: **{user_name}**")

    # Guard against non-persistent sessions (like the 'anon' bypass if handled that way)
    if user_id == 9999:
        st.warning("History is not saved for temporary sessions. Please log in with a registered account to persist data.")
        return

    # Fetch records from the database logic built in Block 2
    # If the function doesn't exist (e.g., during testing), we return an empty list
    records = fetch_history(user_id) if 'fetch_history' in globals() else []

    if not records:
        st.warning("No diagnosis history found. Records appear here after an analysis is completed and saved.")
        return

    # --- VOICE SEARCH UI ---
    st.subheader("Search Records")
    col_search, col_voice = st.columns([4, 1])
    
    with col_search:
        # The placeholder is the "hook" for the JavaScript voice injection
        search_query = st.text_input(
            "Filter by patient, disease, or result", 
            key="voice_search_input", 
            placeholder="Type or use microphone..."
        )
    
    with col_voice:
        st.write(" ") # Alignment spacer
        # Speech-to-Text Bridge: Injects transcribed text directly into the Streamlit input DOM via the parent window
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
                    
                    // Logic to find the Streamlit input field in the parent DOM
                    const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                    let targetInput = null;

                    for (let input of inputs) {
                        if (input.placeholder === "Type or use microphone...") {
                            targetInput = input;
                            break;
                        }
                    }

                    if (targetInput) {
                        // Native value setter to bypass React's internal state block
                        const nativeValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                        nativeValueSetter.call(targetInput, voiceText);

                        // Trigger change detection so Streamlit picks up the new value
                        const event = new Event('input', { bubbles: true });
                        targetInput.dispatchEvent(event);
                        
                        targetInput.focus();
                        setTimeout(() => { targetInput.blur(); }, 100);
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
        <button onclick="startDictation()" style="background-color: #ff4b4b; color: white; border: none; padding: 10px; border-radius: 8px; cursor: pointer; width: 100%; height: 45px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px; transition: 0.3s;">
            🎤 Speak
        </button>
        """
        components.html(voice_js, height=50)

    # --- DATA PROCESSING & FILTERING ---
    history_data = []
    for record in records:
        # Standardize date formatting
        diag_date = record['diagnosis_date']
        date_str = diag_date.strftime('%Y-%m-%d %H:%M') if hasattr(diag_date, 'strftime') else str(diag_date)
        
        history_data.append({
            "Date": date_str,
            "Patient": record['patient_name'],
            "Sample": record['sample_type'],
            "Disease": record['disease_tested'],
            "Result": record['result_status'],
            "Confidence": f"{record['confidence_score']*100:.1f}%",
        })
    
    df = pd.DataFrame(history_data)

    # Real-time search filtering
    if search_query:
        search_query = search_query.lower()
        df = df[
            df['Patient'].str.lower().contains(search_query) | 
            df['Disease'].str.lower().contains(search_query) |
            df['Result'].str.lower().contains(search_query)
        ]

    st.subheader(f"History Log ({len(df)} records)")
    st.dataframe(
        df, 
        use_container_width=True,
        hide_index=True 
    )

def main():
    """Main Application Entry Point & Router."""
    # Ensure set_page_config is the first Streamlit command called
    try:
        st.set_page_config(
            page_title="Pathoscope AI",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except:
        pass # Handle cases where this might be called twice during development
    
    # Initialize Global Auth State
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # --- Sidebar Navigation ---
    st.sidebar.markdown("# 🔬 Pathoscope AI")
    st.sidebar.caption("v1.2.0 | Precision Diagnostics")
    st.sidebar.divider()

    if not st.session_state['logged_in']:
        # If not authenticated, force Login Page (from Block 4)
        page_login()
    else:
        # Sidebar Status & Navigation
        st.sidebar.success(f"User: {st.session_state['full_name']}")
        
        app_page = st.sidebar.radio(
            "Navigate Application",
            ["Diagnosis Portal", "Clinical History"],
            index=0,
            key="main_nav_radio"
        )
        
        st.sidebar.divider()
        
        # Logout Logic
        if st.sidebar.button("🔒 Secure Logout", use_container_width=True):
            keys_to_clear = ['logged_in', 'user_id', 'full_name', 'analysis_result', 'uploaded_image', 'grad_cam_image']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['logged_in'] = False
            st.rerun()

        st.sidebar.info("Support: help@pathoscope.ai")

        # --- Router Execution ---
        if app_page == "Diagnosis Portal":
            page_diagnosis() # Defined in Block 4
        elif app_page == "Clinical History":
            page_history() # Defined in Block 5

if __name__ == '__main__':
    main()
