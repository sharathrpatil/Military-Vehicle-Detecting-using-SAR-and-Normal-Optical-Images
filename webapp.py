import streamlit as st
import pandas as pd
import geocoder
import os
from PIL import Image
from mail import send_email
from ultralytics import YOLO
from pothole_detection import detect_from_image, detect_from_video
import streamlit.components.v1 as components
import subprocess

os.makedirs('uploads', exist_ok=True) # create uploads directory if missing

# -----------------------------
# Model selection (Normal vs SAR)
# -----------------------------
MODEL_PATHS = {
    "Normal Model": r"C:\SUHAS_P\defence pro\ship.pt",
    "SAR Model": r"C:\SUHAS_P\defence pro\aircraft.pt"  # TODO: update with actual SAR model path
}


@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)


# Sidebar menu
page = st.sidebar.selectbox("Pages Menu", options=['Home', 'Using Image', 'Using Video', 'Live Camera'])

# Function to register pothole information
def register(location, highway_type, size, position, is_video=False):
    data = {"location": location, "highway_type": highway_type, "size": size, "position": position}
    send_email(data, 'suhastms2004@gmail.com', is_video)  
    st.info("Reported successfully.")

# Geolocation (fallback to IP-based geolocation)
def get_fallback_location():
    g = geocoder.ip('me') #get the device's public IP address
    if g.latlng:
        return g.latlng
    else:
        return [0, 0]  # Default location

# Function to display map and gather pothole information
def get_pothole_info():
    location = get_fallback_location()
    st.sidebar.markdown("---")

    
    df = pd.DataFrame([location], columns=['lat', 'lon'])
    st.info(f"Location: Latitude {location[0]}, Longitude {location[1]}")

    # Gather additional information about the pothole
    highway_type = st.sidebar.selectbox("Select Road Type:", options=["National Highway", "Local Road"])
    size = st.sidebar.selectbox("Approx. Size of Pothole", options=["Small Pothole", "Medium Pothole", "Large Pothole"])
    position = st.sidebar.selectbox("Position of Pothole", options=["Center", "Sideways"])
    
    return location, highway_type, size, position

# Function to load and save uploaded image
def load_image(image_file):
    img = Image.open(image_file)
    img.save("uploads/image.jpg")
    return img

# Function to load and save uploaded video
def load_video(video_file):
    path_name = "uploads/video.mp4"
    with open(path_name, 'wb') as f:
        f.write(video_file.read()) #read and write in the binary formate to specified path

# Page: Using Image
if page == 'Using Image':
    st.title("Pothole Detection Using Image")
    # Model choice specific to Image detection
    model_choice = st.radio("Select Model", list(MODEL_PATHS.keys()), horizontal=True)
    selected_model_path = MODEL_PATHS[model_choice]
    model = load_model(selected_model_path)
    choice_upload = st.sidebar.selectbox("Select a Method", options=['Upload Image', 'Open Camera'])
    if choice_upload == 'Upload Image':
        image_file = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'])
        if image_file is not None:
            col1, col2 = st.columns(2)
            file_details = {"filename": image_file.name, "filetype": image_file.type, "filesize": image_file.size}
            st.write(file_details)
            col1.image(load_image(image_file))
            detect_from_image("uploads/image.jpg", model)  # Pass the YOLOv8 model here
            col2.image("results/image_result.jpg")
            location, highway_type, size, position = get_pothole_info()
            if st.sidebar.button("Submit Report"):
                register(location, highway_type, size, position)

    elif choice_upload == 'Open Camera':
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            img = Image.open(img_file_buffer)
            img.save("uploads/image.jpg")
            detect_from_image("uploads/image.jpg", model)  # Pass the YOLOv8 model here
            st.image("results/image_result.jpg")
            location, highway_type, size, position = get_pothole_info()
            if st.sidebar.button("Submit Report"):
                register(location, highway_type, size, position)

# Page: Using Video
elif page == 'Using Video':
    st.title("Pothole Detection Using Video")
    # Model choice specific to Video detection
    model_choice = st.radio("Select Model", list(MODEL_PATHS.keys()), horizontal=True)
    selected_model_path = MODEL_PATHS[model_choice]
    model = load_model(selected_model_path)
    video_option = st.sidebar.selectbox("Select a Method", options=["Upload Video", "Live Video"])

    if video_option == "Upload Video":
        video_file = st.file_uploader("Upload Video", type=["mp4", "mkv", "avi"])
        if video_file is not None:
            load_video(video_file)  # here the video get uploaded to the uploads
            detect_from_video("uploads/video.mp4", model)  # Pass the YOLOv8 model here
            os.system('ffmpeg -i results/video_result.avi -vcodec libx264 results/processed.mp4 -y')
            st.snow()
            video_result = open("results/processed.mp4", 'rb')
            video_bytes = video_result.read()
            st.video(video_bytes)
            location, highway_type, size, position = get_pothole_info()
            if st.sidebar.button("Submit Report"):
                register(location, highway_type, size, position, is_video=True)

    elif video_option == "Live Video":
        st.warning("Provide the name of the video file to use for live processing.")
        video_file_name = st.text_input("Enter the video file name (e.g., 'my_video.mp4')", "")

        if st.button("Save Video Source"):
            if video_file_name.strip():
                os.makedirs("config", exist_ok=True)
                with open("config/live_video_src.txt", "w") as f:
                    f.write(video_file_name.strip())
                st.success(f"Video source saved as '{video_file_name.strip()}'.")
            else:
                st.error("Please enter a valid video file name.")

        st.warning("Start or stop detection using the buttons below.")
        start_detection = st.button("Start Detection")
        stop_detection = st.button("Stop Detection")

        if 'process' not in st.session_state:
            st.session_state.process = None

        if start_detection:
            if st.session_state.process is None or st.session_state.process.poll() is not None:
                try:
                    st.session_state.process = subprocess.Popen(
                        ["python", "test1.py", selected_model_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    st.info("Detection started! Check logs for updates.")
                except Exception as e:
                    st.error(f"Error starting process: {e}")
            else:
                st.warning("Detection is already running.")

        if stop_detection:
            if st.session_state.process is not None and st.session_state.process.poll() is None:
                st.session_state.process.terminate()
                st.session_state.process = None
                st.success("Detection stopped successfully!")
            else:
                st.warning("No detection process is currently running.")

        # Display script output if running
        if st.session_state.process is not None and st.session_state.process.poll() is None:
            st.text("Script Output:")
            try:
                output = st.session_state.process.stdout.readline()
                while output:
                    st.text(output.strip())
                    output = st.session_state.process.stdout.readline()
            except Exception as e:
                st.error(f"Error reading process output: {e}")

# Page: Live Camera
elif page == 'Live Camera':
    st.title("Pothole Detection Using Live Camera")
    # Model choice specific to Live Camera
    model_choice = st.radio("Select Model", list(MODEL_PATHS.keys()), horizontal=True)
    selected_model_path = MODEL_PATHS[model_choice]
    model = load_model(selected_model_path)
    st.warning("Running the external script test.py for live video detection. This might take some time.")
    start_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")

    if 'process' not in st.session_state:
        st.session_state.process = None

    if start_button:
        if st.session_state.process is None or st.session_state.process.poll() is not None:
            try:
                st.session_state.process = subprocess.Popen(
                    ["python", "test.py", selected_model_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                st.info("Script started! Check logs for updates.")
            except Exception as e:
                st.error(f"Error starting script: {e}")
        else:
            st.warning("Script is already running!")

    if stop_button:
        if st.session_state.process is not None and st.session_state.process.poll() is None:
            st.session_state.process.terminate()
            st.session_state.process = None
            st.success("Script stopped successfully!")
        else:
            st.warning("No script is currently running.")

    if st.session_state.process is not None and st.session_state.process.poll() is None:
        st.text("Script Output:")
        try:
            output = st.session_state.process.stdout.readline()
            while output:
                st.text(output.strip())
                output = st.session_state.process.stdout.readline()
        except Exception as e:
            st.error(f"Error reading script output: {e}")

# Page: Home
else:
    # ------------------------------
    # Stylish Home / Landing Section
    # ------------------------------
    
    # Custom BG & font styles
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            }
            h1, h2, h3, h4, h5, h6, p {
                color: #ffffff !important;
            }
            .feature-card {
                padding: 1rem;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 0.5rem;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1 style='text-align:center;font-size:3rem;'>🛡️ Military Defense Detection System</h1>", unsafe_allow_html=True)
    st.image("image1.jpeg", use_column_width=True)
    st.markdown("<p style='text-align:center;font-size:1.2rem;'>Ensuring mission readiness through rapid terrain and target analysis</p>", unsafe_allow_html=True)
    # Navigation bar
    st.markdown("""<div style='display:flex;justify-content:center;gap:1.5rem;font-size:1.1rem;margin:1rem 0;'>
<a style='color:white;text-decoration:none;' href='#'>Home</a>
<a style='color:white;text-decoration:none;' href='#'>Detect Image</a>
<a style='color:white;text-decoration:none;' href='#'>Detect Video</a>
<a style='color:white;text-decoration:none;' href='#'>Live Camera</a>
<a style='color:white;text-decoration:none;' href='#'>Reports</a>
    </div>""", unsafe_allow_html=True)

    # Defense context explanation
    st.markdown("""### Why Detection Matters for Defence
Timely identification of potholes, hidden threats or unexploded ordnance is critical for convoy safety and strategic mobility. Synthetic Aperture Radar (SAR) images allow all-weather, night-time surveillance while normal optical imagery provides high-resolution daytime visuals.
""")

    st.markdown("## Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='feature-card'>📷<br><b>Image Detection</b></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='feature-card'>🎥<br><b>Video Detection</b></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='feature-card'>📡<br><b>Live Camera</b></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;'>Powered by YOLOv8 & Streamlit • Developed with ❤️ for safer roads</p>",
        unsafe_allow_html=True,
    )

    st.balloons()
