from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import torch
from PIL import Image
import os
from werkzeug.security import generate_password_hash, check_password_hash
import torchvision.transforms as transforms

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path for saving uploads
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Define your model architecture
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 512)
        self.fc2 = torch.nn.Linear(512, 8)  # Assuming 8 classes for blood groups

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and load the state dictionary
model = SimpleCNN()
model.load_state_dict(torch.load('fingerprint_blood_group_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation for the input image
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict blood group route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save file to upload folder
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load image, apply transformations
        img = Image.open(file_path).convert('RGB')
        img_tensor = data_transform(img).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            prediction = model(img_tensor)

        predicted_blood_group = torch.argmax(prediction).item()
        blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        predicted_group_name = blood_groups[predicted_blood_group]

        # Redirect to result page with the predicted blood group
        return render_template('resultpage.html', blood_group=predicted_group_name)

# Routes for different pages
@app.route('/')
def home():
    return render_template('loader.html')

@app.route('/homepage')
def mainPage():
    return render_template('homepage.html')

@app.route('/test')
def testpage():
    return render_template('testpage.html')

@app.route('/result')
def resultpage():
    return render_template('resultpage.html')

if __name__ == '__main__':
    app.run(debug=True)
