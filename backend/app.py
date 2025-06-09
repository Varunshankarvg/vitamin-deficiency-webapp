from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from torchcam.methods import SmoothGradCAMpp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime
import os
from pymongo import MongoClient
import gridfs
import io

# Setup Flask app
app = Flask(__name__)
CORS(app)

# MongoDB connection setup
MONGO_URI = 'mongodb+srv://targaryen:friends@miniproject.fptrr7c.mongodb.net/'
client = MongoClient(MONGO_URI)
db = client['targaryen']
fs = gridfs.GridFS(db)

# Load your model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = [
    'Vitamin A_C Deficiency',
    'Vitamin B12_E Deficiency',
    'Vitamin B2_B3 Deficiency',
    'Vitamin D_Biotin Deficiency'
]
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(torch.nn.Linear(num_ftrs, len(class_names)))
model.load_state_dict(torch.load('best_vitamin_model2.pth', map_location=device))
model = model.to(device)
model.eval()

# Helper dictionaries
food_suggestions = {
    "Vitamin A_C Deficiency": "Eat carrots, sweet potatoes, spinach, oranges, broccoli.",
    "Vitamin D_Biotin Deficiency": "Get daily sunlight, eat fatty fish (salmon, tuna), egg yolk, fortified milk.",
    "Vitamin B2_B3 Deficiency": "Eat leafy greens, nuts, dairy products, poultry, whole grains.",
    "Vitamin B12_E Deficiency": "Eat meat, fish, dairy products, fortified cereals, or take supplements."
}

medical_recommendations = {
    "Mild": "Focus on improving diet. Monitor skin condition and recheck after 30 days.",
    "Moderate": "Consult a doctor for supplements and advice. Monitor for symptoms.",
    "Severe": "Immediate consultation with dermatologist strongly recommended."
}

vitamin_info = {
    "Vitamin A_C Deficiency": {
        "Function": "Maintains healthy skin, vision, and immune function.",
        "Symptoms": "Dry skin, night blindness, frequent infections."
    },
    "Vitamin D_Biotin Deficiency": {
        "Function": "Supports bone health, immune strength, and skin healing.",
        "Symptoms": "Fatigue, bone pain, hair loss, muscle weakness."
    },
    "Vitamin B2_B3 Deficiency": {
        "Function": "Helps energy production, skin health, and digestion.",
        "Symptoms": "Skin cracks, mouth sores, sensitivity to light."
    },
    "Vitamin B12_E Deficiency": {
        "Function": "Helps red blood cell formation, nerve function, DNA synthesis.",
        "Symptoms": "Weakness, numbness, memory loss, mood changes."
    }
}

# Helper function
def run_model_and_gradcam(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    cam_extractor = SmoothGradCAMpp(model, target_layer=model.features[-1])
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1).detach().cpu().numpy().squeeze()

    predicted_labels = []
    confidence_threshold = 0.5
    for idx, prob in enumerate(probs):
        if prob > confidence_threshold:
            predicted_labels.append((class_names[idx], prob))
    if not predicted_labels:
        top_idx = np.argmax(probs)
        predicted_labels = [(class_names[top_idx], probs[top_idx])]

    predicted_class = predicted_labels[0][0]
    activation_map = cam_extractor(class_names.index(predicted_class), outputs)[0].squeeze(0).cpu().numpy()
    heatmap = (activation_map - np.min(activation_map)) / (np.max(activation_map) - np.min(activation_map))
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    activated_pixels = np.sum(heatmap_resized > 0.5)
    severity_percent = (activated_pixels / heatmap_resized.size) * 100
    severity_level = "Mild" if severity_percent <= 20 else "Moderate" if severity_percent <= 50 else "Severe"

    # Save GradCAM image
    os.makedirs('static', exist_ok=True)
    input_image = np.array(image.resize((224, 224))).astype(np.float32) / 255
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET) / 255
    blended_image = (input_image * 0.5 + heatmap_colored * 0.5)
    blended_image = np.clip(blended_image, 0, 1)
    gradcam_path = os.path.join('static', 'gradcam_result.png')
    plt.imsave(gradcam_path, blended_image)

    return predicted_labels, severity_percent, severity_level, gradcam_path

# PDF Report Generator
def predict_and_generate(image_path):
    predicted_labels, severity_percent, severity_level, gradcam_path = run_model_and_gradcam(image_path)

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "Vitamin Deficiency Skin Analysis Report", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Date: {datetime.datetime.now().strftime('%d %B %Y')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Predicted Deficiencies:", ln=True)
    pdf.set_font("Arial", '', 12)
    for label, prob in predicted_labels:
        pdf.cell(0, 8, f"- {label}: {prob*100:.2f}%", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Severity Analysis:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Affected Area: {severity_percent:.2f}%", ln=True)
    pdf.cell(0, 8, f"Severity Level: {severity_level}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Medical Recommendations:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, medical_recommendations.get(severity_level, "Consult a doctor."))
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Food & Diet Suggestions:", ln=True)
    pdf.set_font("Arial", '', 12)
    for label, _ in predicted_labels:
        foods = food_suggestions.get(label, "Maintain healthy diet.")
        pdf.multi_cell(0, 8, f"For {label}: {foods}")
    pdf.ln(5)

    pdf.image(gradcam_path, x=10, w=190)
    pdf_path = 'report.pdf'
    pdf.output(pdf_path)

    # Upload files to MongoDB
    with open(image_path, 'rb') as f:
        original_image_id = fs.put(f, filename=os.path.basename(image_path))
    with open(gradcam_path, 'rb') as f:
        gradcam_image_id = fs.put(f, filename='gradcam_result.png')
    with open(pdf_path, 'rb') as f:
        pdf_report_id = fs.put(f, filename='report.pdf')

    db.reports.insert_one({
        "filename": os.path.basename(image_path),
        "predictions": [{"label": label, "confidence": float(conf)} for label, conf in predicted_labels],
        "severity_percent": severity_percent,
        "severity_level": severity_level,
        "upload_time": datetime.datetime.utcnow(),
        "original_image_id": original_image_id,
        "gradcam_image_id": gradcam_image_id,
        "pdf_report_id": pdf_report_id
    })

    return pdf_path

# === ROUTES ===

@app.route('/predict', methods=['POST'])
def predict_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    report_path = predict_and_generate(filepath)

    # Send file first
    response = send_file(report_path, as_attachment=True)

    # Then delete
    try:
        os.remove(filepath)
        os.remove('static/gradcam_result.png')
        os.remove(report_path)
    except Exception as e:
        print(f"Error deleting files: {e}")

    return response

@app.route('/predict-details', methods=['POST'])
def predict_json():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    predicted_labels, severity_percent, severity_level, _ = run_model_and_gradcam(filepath)
    return jsonify({
        'predictions': [
            {"label": label, "confidence": float(conf)} for label, conf in predicted_labels
        ],
        'severity_percent': severity_percent,
        'severity_level': severity_level,
        'gradcam_url': '/gradcam'
    })

@app.route('/gradcam')
def gradcam():
    return send_from_directory('static', 'gradcam_result.png')

# === START APP ===
if __name__ == '__main__':
    app.run(debug=True)
