from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import random
import uuid
from datetime import datetime
import math
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64
import logging
import traceback
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mango Tree Location API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class LocationData(BaseModel):
    imageName: str
    latitude: float
    longitude: float
    imageId: str

class LocationRequest(BaseModel):
    locations: List[LocationData]

class SimilarPair(BaseModel):
    pairId: str
    imageId1: str
    imageId2: str
    distance: float
    imageName1: str
    imageName2: str

class ProximityResponse(BaseModel):
    similar_pairs: List[SimilarPair]
    total_images: int
    pairs_found: int

class DecisionRequest(BaseModel):
    pairId: str
    action: str  # 'save_both', 'keep_first_remove_second', 'remove_first_keep_second'
    imageId1: str
    imageId2: str

class DecisionResponse(BaseModel):
    success: bool
    message: str
    saved_to_database: bool

class ImageClassificationRequest(BaseModel):
    image_data: str  # base64 encoded image
    model_type: str  # 'teachable_machine' or 'mobilenet'

class ClassificationResult(BaseModel):
    className: str
    probability: float

class ImageClassificationResponse(BaseModel):
    predictions: List[ClassificationResult]
    model_used: str

# Global variables
mobilenet_model = None
MOBILENET_CLASSES = ['mango_tree', 'not_mango_tree']  # Adjust according to your model
processed_locations = []
decisions_log = []

# Load the fine-tuned model with detailed error handling
def load_mobilenet_model():
    global mobilenet_model
    try:
        model_path = 'mango_model-v001.h5'
        logger.info(f"Attempting to load model from: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Files in current directory: {os.listdir('.')}")
            return False
        
        # Check file size
        file_size = os.path.getsize(model_path)
        logger.info(f"Model file size: {file_size} bytes")
        
        # Load model
        mobilenet_model = tf.keras.models.load_model(model_path)
        logger.info("MobileNetV2 model loaded successfully")
        
        # Log model information
        logger.info(f"Model input shape: {mobilenet_model.input_shape}")
        logger.info(f"Model output shape: {mobilenet_model.output_shape}")
        logger.info(f"Number of layers: {len(mobilenet_model.layers)}")
        
        # Test model with dummy input
        dummy_input = np.random.random((1, 224, 224, 3))
        test_prediction = mobilenet_model.predict(dummy_input)
        logger.info(f"Test prediction shape: {test_prediction.shape}")
        logger.info(f"Test prediction values: {test_prediction}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading MobileNetV2 model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        mobilenet_model = None
        return False

# Initialize model on startup
model_loaded = load_mobilenet_model()
logger.info(f"Model initialization result: {model_loaded}")

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the distance between two GPS coordinates using Haversine formula.
    Returns distance in meters.
    """
    # TODO: Implement actual distance calculation logic
    # For now, return a mock distance for testing
    
    # Radius of the Earth in meters
    R = 6371000  
    
    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in meters
    distance = R * c
    return distance

def find_nearby_pairs(locations: List[LocationData], threshold_meters: float = 1.0) -> List[SimilarPair]:
    """
    Find pairs of images that are within threshold distance of each other.
    """
    # TODO: Implement actual proximity checking logic
    # For now, return 2 random pairs for frontend testing
    
    if len(locations) < 2:
        return []
    
    pairs = []
    
    # Create 2 random pairs for testing
    for i in range(min(2, len(locations) // 2)):
        if i * 2 + 1 < len(locations):
            loc1 = locations[i * 2]
            loc2 = locations[i * 2 + 1]
            
            distance = calculate_distance(
                loc1.latitude, loc1.longitude,
                loc2.latitude, loc2.longitude
            )
            
            pair = SimilarPair(
                pairId=str(uuid.uuid4()),
                imageId1=loc1.imageId,
                imageId2=loc2.imageId,
                distance=distance,
                imageName1=loc1.imageName,
                imageName2=loc2.imageName
            )
            pairs.append(pair)
    
    return pairs

def preprocess_image_for_mobilenet(image_data: str) -> np.ndarray:
    """
    Preprocess base64 image data for MobileNetV2 model.
    """
    try:
        logger.info("Starting image preprocessing")
        logger.info(f"Image data length: {len(image_data)}")
        
        # Decode base64 image
        if ',' in image_data:
            header, encoded = image_data.split(',', 1)
            logger.info(f"Image header: {header}")
            image_bytes = base64.b64decode(encoded)
        else:
            image_bytes = base64.b64decode(image_data)
        
        logger.info(f"Decoded image bytes length: {len(image_bytes)}")
        
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Original image size: {image.size}")
        logger.info(f"Original image mode: {image.mode}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info("Converted image to RGB")
        
        # Resize to MobileNetV2 input size (224x224)
        image = image.resize((224, 224))
        logger.info(f"Resized image to: {image.size}")
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        logger.info(f"Image array shape before normalization: {image_array.shape}")
        logger.info(f"Image array dtype: {image_array.dtype}")
        logger.info(f"Image array min/max values: {image_array.min()}/{image_array.max()}")
        
        # Normalize to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        logger.info(f"Image array min/max after normalization: {image_array.min()}/{image_array.max()}")
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        logger.info(f"Final image array shape: {image_array.shape}")
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def classify_with_mobilenet(image_array: np.ndarray) -> List[ClassificationResult]:
    """
    Classify image using the fine-tuned MobileNetV2 model.
    """
    try:
        logger.info("Starting MobileNet classification")
        
        if mobilenet_model is None:
            logger.error("MobileNetV2 model is not loaded")
            raise HTTPException(status_code=500, detail="MobileNetV2 model not loaded")
        
        logger.info(f"Input array shape: {image_array.shape}")
        logger.info(f"Input array dtype: {image_array.dtype}")
        
        # Make prediction
        logger.info("Making prediction...")
        predictions = mobilenet_model.predict(image_array, verbose=0)
        logger.info(f"Raw predictions shape: {predictions.shape}")
        logger.info(f"Raw predictions: {predictions}")
        
        # Convert predictions to result format
        results = []
        for i, prob in enumerate(predictions[0]):
            className = MOBILENET_CLASSES[i] if i < len(MOBILENET_CLASSES) else f"class_{i}"
            result = ClassificationResult(
                className=className,
                probability=float(prob)
            )
            results.append(result)
            logger.info(f"Class {className}: {float(prob):.4f}")
        
        # Sort by probability (highest first)
        results.sort(key=lambda x: x.probability, reverse=True)
        
        logger.info("Classification completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in MobileNet classification: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Mango Tree Location API is running", 
        "timestamp": datetime.now(),
        "model_loaded": mobilenet_model is not None,
        "tensorflow_version": tf.__version__
    }

@app.get("/model-status")
async def model_status():
    """Get detailed model status"""
    return {
        "model_loaded": mobilenet_model is not None,
        "model_classes": MOBILENET_CLASSES,
        "tensorflow_version": tf.__version__,
        "model_input_shape": mobilenet_model.input_shape if mobilenet_model else None,
        "model_output_shape": mobilenet_model.output_shape if mobilenet_model else None,
        "current_directory": os.getcwd(),
        "model_file_exists": os.path.exists('mango_model-v001.h5')
    }

@app.post("/reload-model")
async def reload_model():
    """Reload the MobileNet model"""
    success = load_mobilenet_model()
    return {
        "success": success,
        "model_loaded": mobilenet_model is not None,
        "message": "Model reloaded successfully" if success else "Failed to reload model"
    }

@app.post("/check-proximity", response_model=ProximityResponse)
async def check_proximity(request: LocationRequest):
    """
    Receive locations and check if any two mango tree images are less than 1m apart.
    Returns similar image pairs/groups.
    """
    try:
        logger.info(f"Checking proximity for {len(request.locations)} locations")
        
        # Store locations globally
        global processed_locations
        processed_locations.extend(request.locations)
        
        # Find nearby pairs
        similar_pairs = find_nearby_pairs(request.locations)
        
        logger.info(f"Found {len(similar_pairs)} similar pairs")
        
        return ProximityResponse(
            similar_pairs=similar_pairs,
            total_images=len(request.locations),
            pairs_found=len(similar_pairs)
        )
        
    except Exception as e:
        logger.error(f"Error in check_proximity: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing locations: {str(e)}")

@app.post("/save-decision", response_model=DecisionResponse)
async def save_decision(request: DecisionRequest):
    """
    Save user's decision about duplicate image pairs to database.
    """
    try:
        logger.info(f"Saving decision: {request.action} for pair {request.pairId}")
        
        global decisions_log
        
        decision_record = {
            "id": str(uuid.uuid4()),
            "pairId": request.pairId,
            "action": request.action,
            "imageId1": request.imageId1,
            "imageId2": request.imageId2,
            "timestamp": datetime.now(),
            "saved_to_db": True  # Mock success
        }
        
        decisions_log.append(decision_record)
        
        return DecisionResponse(
            success=True,
            message=f"Decision '{request.action}' saved successfully",
            saved_to_database=True
        )
        
    except Exception as e:
        logger.error(f"Error in save_decision: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error saving decision: {str(e)}")

@app.post("/classify-image", response_model=ImageClassificationResponse)
async def classify_image(request: ImageClassificationRequest):
    """
    Classify image using either Teachable Machine or MobileNetV2 model.
    """
    try:
        logger.info(f"Classification request received with model type: {request.model_type}")
        
        if request.model_type == 'mobilenet':
            logger.info("Processing with MobileNetV2 model")
            
            # Use MobileNetV2 model
            image_array = preprocess_image_for_mobilenet(request.image_data)
            predictions = classify_with_mobilenet(image_array)
            
            response = ImageClassificationResponse(
                predictions=predictions,
                model_used='mobilenet'
            )
            
            logger.info(f"Classification successful, returning {len(predictions)} predictions")
            return response
            
        else:
            # Return error for teachable machine as it should be handled on frontend
            logger.warning(f"Invalid model type requested: {request.model_type}")
            raise HTTPException(
                status_code=400, 
                detail="Teachable Machine classification should be handled on frontend"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in classify_image: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error classifying image: {str(e)}")

@app.get("/decisions")
async def get_decisions():
    """
    Get all saved decisions (for debugging/admin purposes).
    """
    return {
        "decisions": decisions_log,
        "total_decisions": len(decisions_log)
    }

@app.get("/locations")
async def get_locations():
    """
    Get all processed locations (for debugging/admin purposes).
    """
    return {
        "locations": processed_locations,
        "total_locations": len(processed_locations)
    }

@app.delete("/clear-data")
async def clear_data():
    """
    Clear all stored data (for testing purposes).
    """
    global processed_locations, decisions_log
    processed_locations.clear()
    decisions_log.clear()
    
    return {"message": "All data cleared successfully"}

@app.post("/bulk-save")
async def bulk_save_images():
    """
    TODO: Endpoint to bulk save approved images to permanent storage.
    """
    # Implement logic to save approved images to database/cloud storage
    return {"message": "Bulk save endpoint - TODO: Implement actual logic"}

@app.get("/statistics")
async def get_statistics():
    """
    Get processing statistics.
    """
    total_locations = len(processed_locations)
    total_decisions = len(decisions_log)
    
    # Count decision types
    save_both = sum(1 for d in decisions_log if d["action"] == "save_both")
    keep_first = sum(1 for d in decisions_log if d["action"] == "keep_first_remove_second")
    keep_second = sum(1 for d in decisions_log if d["action"] == "remove_first_keep_second")
    
    return {
        "total_locations_processed": total_locations,
        "total_decisions_made": total_decisions,
        "decisions_breakdown": {
            "save_both": save_both,
            "keep_first_remove_second": keep_first,
            "remove_first_keep_second": keep_second
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")