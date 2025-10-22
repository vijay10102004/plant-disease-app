import streamlit as st 
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import google.generativeai as genai  # ✅ Gemini Integration
import time
import re

# -----------------------------
# CONFIGURATION
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "plantDisease-resnet34.pth"
num_classes = 40

# Hardcoded credentials (demo)
users = {"admin": "admin123"}

# -----------------------------
# CLASS LIST & MAPPING
# -----------------------------
class_list = [
    "AppleCedarRust1", "AppleCedarRust2", "AppleCedarRust3", "AppleCedarRust4",
    "AppleScab1", "AppleScab2", "AppleScab3", "BetelHealthy",
    "CornCommonRust1", "CornCommonRust2", "CornCommonRust3",
    "DriedLeaf1", "DriedLeaf2", "DriedLeaf3",
    "JackfruitBpot1", "JackfruitHealthy1", "JackfruitHealthy2",
    "PotatoEarlyBlight1", "PotatoEarlyBlight2", "PotatoEarlyBlight3",
    "PotatoEarlyBlight4", "PotatoEarlyBlight5",
    "PotatoHealthy1", "PotatoHealthy2",
    "TomatoEarlyBlight1", "TomatoEarlyBlight2", "TomatoEarlyBlight3",
    "TomatoEarlyBlight4", "TomatoEarlyBlight5", "TomatoEarlyBlight6",
    "TomatoHealthy1", "TomatoHealthy2", "TomatoHealthy3", "TomatoHealthy4",
    "TomatoYellowCurlVirus1", "TomatoYellowCurlVirus2", "TomatoYellowCurlVirus3",
    "TomatoYellowCurlVirus4", "TomatoYellowCurlVirus5", "TomatoYellowCurlVirus6"
]

filename_to_label = {f"{name}.JPG": name for name in class_list}

# -----------------------------
# IMAGE TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Checkpoint not found at '{checkpoint_path}'")
        st.stop()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        st.success("✅ Model loaded successfully (strict mode).")
    except RuntimeError:
        pretrained_dict = {k: v for k, v in new_state_dict.items() if "fc" not in k}
        model.load_state_dict(pretrained_dict, strict=False)
        st.warning("⚠️ Loaded conv layers; fc layer random.")

    model.eval()
    return model

# -----------------------------
# CLEAN LABEL FUNCTION
# -----------------------------
def clean_label(label):
    """Remove trailing numbers and split CamelCase words."""
    label = re.sub(r'\d+', '', label)  # remove numbers
    label = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)  # split CamelCase
    return label.strip()

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(image: Image.Image, model, img_name=None):
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
    pred_class = class_list[pred_idx.item()]
    pred_label = filename_to_label.get(img_name, pred_class) if img_name else pred_class
    return clean_label(pred_label), conf.item() * 100

# -----------------------------
# AI REMEDY FUNCTION (GEMINI)
# -----------------------------
def get_ai_remedy(disease_name):
    try:
        genai.configure(api_key="AIzaSyCtjW1oim7SmMgDVMuTXKeGkJAKYFtAuE0")  # replace or use env var
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
        You are an expert agronomist.
        Provide short, clear remedies and prevention methods for {disease_name} in crops.
        Use bullet points with emojis for better readability.
        """
        response = model.generate_content(prompt)
        return response.text if response and response.text else "⚠️ No remedy generated. Try again later."
    except Exception as e:
        return f"⚠️ Gemini AI error: {str(e)}"

# -----------------------------
# CUSTOM CSS + ANIMATIONS
# -----------------------------
def set_custom_css():
    st.markdown("""
    <style>
    body { background-color: #eef2f3; color: #333; font-family:'Segoe UI',sans-serif; }
    .card { background-color: #fff; border-radius: 20px; padding:30px; margin:25px 0;
            box-shadow:0 6px 20px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-in-out; transition:transform 0.3s;}
    .card:hover { transform: scale(1.02);}
    .badge { padding:12px 22px; border-radius:25px; color:white; font-weight:bold; font-size:18px;
             display:inline-block; animation: pulse 1.5s infinite;}
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .footer { margin-top:50px; padding:15px; text-align:center; color:#777; font-size:14px;}
    header, footer {visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# LOGIN PAGE
# -----------------------------
def login_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("🔐 AgriSense Pro Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and password == users[username]:
            st.session_state['logged_in'] = True
            st.session_state['login_rerun'] = True
            st.balloons()
        else:
            st.error("❌ Invalid username or password!")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get('login_rerun', False):
        st.session_state['login_rerun'] = False
        st.stop()

# -----------------------------
# DASHBOARD
# -----------------------------
def dashboard():
    st.sidebar.title("🍃 AgriSense Pro")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['login_rerun'] = True
        st.stop()

    model = load_model()
    set_custom_css()

    st.title("🌿 Leaf Disease Detection Dashboard")

    # --- SINGLE IMAGE PREDICTION ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📸 Single Image Prediction")
    uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🧠 Analyzing leaf..."):
            time.sleep(1.5)
            label, confidence = predict_image(image, model, uploaded_file.name)

        badge_color = "#28a745" if "Healthy" in label else "#dc3545"
        st.markdown(f"""
        <h3>Prediction:</h3>
        <span class="badge" style="background-color:{badge_color}">{label}</span>
        <h4>Confidence: {confidence:.2f}%</h4>
        """, unsafe_allow_html=True)

        with st.spinner("🌱 Fetching remedy suggestions..."):
            remedy = get_ai_remedy(label)
            time.sleep(1.2)

        st.success("✅ Remedy generated successfully!")
        st.markdown(f"<b>Recommended Remedy:</b><br>{remedy}", unsafe_allow_html=True)

        st.balloons()

    st.markdown('</div>', unsafe_allow_html=True)

    # --- BATCH PREDICTION ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Batch Prediction (Multiple Images)")
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        results = []
        progress = st.progress(0)
        for i, file in enumerate(uploaded_files):
            img = Image.open(file).convert("RGB")
            label, confidence = predict_image(img, model, file.name)
            results.append({"Image": file.name, "Predicted Disease": label, "Confidence %": f"{confidence:.2f}"})
            progress.progress((i + 1) / len(uploaded_files))
            time.sleep(0.3)
        st.success("✅ Batch processing complete!")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">© 2025 AgriSense Pro | Developed by Vijay N K</div>', unsafe_allow_html=True)

# -----------------------------
# APP ENTRY POINT
# -----------------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
else:
    dashboard()
