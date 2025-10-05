from flask import Flask, render_template, request, jsonify
from Waste_Management_Backend import classify_waste_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_path = "temp.jpg"
    file.save(image_path)

    prediction = classify_waste_image(image_path)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
