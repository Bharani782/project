import os
import pickle
import numpy as np
import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Define upload folder
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the directory exists

# Load the pre-trained model
MODEL_PATH = 'finalized_model.pkl'
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Function to decode prediction into a disease, solution, and link
def decode_prediction(prediction):
    mapping = {
        0: {"disease": "Actinic keratosis", 
            "solution": "Use sunscreen and avoid excessive sun exposure. Consult a dermatologist for cryotherapy or photodynamic therapy.",
            "link": "https://www.cdc.gov/actinic_keratosis"},
        1: {"disease": "Atopic Dermatitis", 
            "solution": "Moisturize regularly, avoid irritants, and use prescribed topical corticosteroids.",
            "link": "https://www.nhs.uk/conditions/atopic-eczema/"},
        2: {"disease": "Benign keratosis", 
            "solution": "No treatment required, but cryotherapy or laser treatment can be done for cosmetic reasons.",
            "link": "https://www.aad.org/public/diseases/keratoses"},
        3: {"disease": "Dermatofibroma", 
            "solution": "No treatment required unless painful. Surgical removal can be an option.",
            "link": "https://www.aad.org/public/diseases/dermatofibromas"},
        4: {"disease": "Melanocytic nevus", 
            "solution": "Most are harmless but regular monitoring is recommended. Seek medical advice if changes occur.",
            "link": "https://www.cancer.org/cancer/melanoma-skin-cancer.html"},
        5: {"disease": "Melanoma", 
            "solution": "Seek immediate medical advice for potential surgery or targeted therapy.",
            "link": "https://www.cancer.gov/types/skin/melanoma"},
        6: {"disease": "Squamous cell carcinoma", 
            "solution": "Seek medical advice for surgical removal, radiation, or topical treatments.",
            "link": "https://www.cdc.gov/squamous_cell_carcinoma"},
        7: {"disease": "Tinea Ringworm Candidiasis", 
            "solution": "Antifungal medications such as creams or oral treatments. Maintain proper hygiene.",
            "link": "https://www.cdc.gov/fungal/diseases/ringworm/index.html"},
        8: {"disease": "Vascular lesion", 
            "solution": "Most are harmless and do not require treatment. Laser therapy is an option for cosmetic reasons.",
            "link": "https://www.aad.org/public/diseases/vascular-lesions"}
    }
    
    prediction_index = np.argmax(prediction)
    result = mapping[prediction_index]
    return result["disease"], result["solution"], result["link"]

# Function to preprocess the uploaded image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image file. Could not read the image.")
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Django view to handle the prediction
@csrf_exempt
def predict_disease(request):
    if request.method == "POST":
        try:
            # Retrieve the uploaded image
            uploaded_file = request.FILES['image']
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            
            # Save the uploaded image
            with open(file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            
            # Preprocess the image
            image = preprocess_image(file_path)
            
            # Perform prediction
            prediction = model.predict(image)
            
            # Decode prediction
            disease, solution, link = decode_prediction(prediction)
            
            # Return the result as a JSON response
            return JsonResponse({
                "disease": disease,
                "solution": solution,
                "link": link
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"message": "Only POST method is allowed."}, status=405)



  