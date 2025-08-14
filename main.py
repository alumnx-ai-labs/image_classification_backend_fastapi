from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
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
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mango Tree Location API", version="2.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Cloudinary (set these environment variables)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dziusjkjr"),
    api_key=os.getenv("CLOUDINARY_API_KEY", "352935158662331"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "PjH1QOK6zTkgIRFQYimq7rN6XvY")
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

# New models from second file
class ImageUploadRequest(BaseModel):
    imageId: str
    imageName: str
    imageData: str  # base64
    latitude: float
    longitude: float
    metadata: Optional[Dict] = None

class CloudStorageResponse(BaseModel):
    success: bool
    imageId: str
    cloudinaryUrl: str
    publicId: str
    message: str

class BulkSaveResponse(BaseModel):
    success: bool
    saved_count: int
    failed_count: int
    saved_images: List[Dict]
    failed_images: List[Dict]
    message: str

# Global variables
mobilenet_model = None
MOBILENET_CLASSES = ['mango_tree', 'not_mango_tree']  # Adjust according to your model
processed_locations = []
decisions_log = []
cloud_storage_records = []  # New from second file

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
    """Calculate distance using Haversine formula"""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    R = 6371000  # Earth radius in meters
    return R * c

def find_nearby_pairs(locations: List[LocationData], threshold_meters: float = 1.0) -> List[SimilarPair]:
    """Find pairs within threshold distance"""
    if len(locations) < 2:
        return []
    
    pairs = []
    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            loc1, loc2 = locations[i], locations[j]
            distance = calculate_distance(loc1.latitude, loc1.longitude, loc2.latitude, loc2.longitude)
            
            if distance <= threshold_meters:
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

# New function from second file
def upload_image_to_cloudinary(image_data: str, image_id: str, metadata: Dict = None) -> Dict:
    """Upload base64 image to Cloudinary"""
    try:
        # Decode base64 image
        if ',' in image_data:
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
        else:
            image_bytes = base64.b64decode(image_data)

        # Create public_id for Cloudinary
        public_id = f"mango_trees/{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            image_bytes,
            public_id=public_id,
            folder="mango_trees",
            resource_type="image",
            context=metadata or {},
            tags=["mango_tree", "gps_data"]
        )
        
        return {
            "success": True,
            "url": upload_result["secure_url"],
            "public_id": upload_result["public_id"],
            "cloudinary_id": upload_result["public_id"],
            "size": upload_result.get("bytes", 0),
            "format": upload_result.get("format", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Cloudinary upload failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# New function from second file
def get_approved_images() -> List[str]:
    """Get list of approved image IDs from decisions"""
    approved = set()
    
    for decision in decisions_log:
        if decision["action"] == "save_both":
            approved.add(decision["imageId1"])
            approved.add(decision["imageId2"])
        elif decision["action"] == "keep_first_remove_second":
            approved.add(decision["imageId1"])
        elif decision["action"] == "remove_first_keep_second":
            approved.add(decision["imageId2"])
    
    return list(approved)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Mango Tree Location API with MobileNet and Cloudinary Storage", 
        "version": "2.0.0",
        "timestamp": datetime.now(),
        "model_loaded": mobilenet_model is not None,
        "tensorflow_version": tf.__version__,
        "cloudinary_configured": bool(os.getenv("CLOUDINARY_CLOUD_NAME"))
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

# New endpoint from second file
@app.post("/upload-image", response_model=CloudStorageResponse)
async def upload_single_image(request: ImageUploadRequest):
    """Upload single image to Cloudinary"""
    try:
        metadata = {
            "imageId": request.imageId,
            "imageName": request.imageName,
            "latitude": str(request.latitude),
            "longitude": str(request.longitude),
            "upload_timestamp": datetime.now().isoformat()
        }
        
        if request.metadata:
            metadata.update(request.metadata)
        
        result = upload_image_to_cloudinary(request.imageData, request.imageId, metadata)
        
        if result["success"]:
            # Store record locally
            cloud_record = {
                "imageId": request.imageId,
                "imageName": request.imageName,
                "cloudinaryUrl": result["url"],
                "publicId": result["public_id"],
                "latitude": request.latitude,
                "longitude": request.longitude,
                "uploadedAt": datetime.now(),
                "size": result.get("size", 0),
                "format": result.get("format", "unknown")
            }
            cloud_storage_records.append(cloud_record)
            
            return CloudStorageResponse(
                success=True,
                imageId=request.imageId,
                cloudinaryUrl=result["url"],
                publicId=result["public_id"],
                message="Image uploaded successfully to Cloudinary"
            )
        else:
            raise HTTPException(status_code=500, detail=f"Upload failed: {result['error']}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk-save", response_model=BulkSaveResponse)
async def bulk_save_images():
    """Bulk save approved images to Cloudinary"""
    try:
        approved_image_ids = get_approved_images()
        
        if not approved_image_ids:
            return BulkSaveResponse(
                success=True,
                saved_count=0,
                failed_count=0,
                saved_images=[],
                failed_images=[],
                message="No approved images to save"
            )
        
        # Find approved locations
        approved_locations = [
            loc for loc in processed_locations 
            if loc.imageId in approved_image_ids
        ]
        
        saved_images = []
        failed_images = []
        
        for location in approved_locations:
            # Check if already uploaded
            existing = next((r for r in cloud_storage_records if r["imageId"] == location.imageId), None)
            if existing:
                saved_images.append({
                    "imageId": location.imageId,
                    "imageName": location.imageName,
                    "cloudinaryUrl": existing["cloudinaryUrl"],
                    "status": "already_uploaded"
                })
                continue
            
            # Mock image data for demo (in real app, you'd have the actual image data)
            # You would typically store image data or file paths with locations
            mock_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            
            try:
                metadata = {
                    "imageId": location.imageId,
                    "imageName": location.imageName,
                    "latitude": str(location.latitude),
                    "longitude": str(location.longitude),
                    "bulk_save_timestamp": datetime.now().isoformat()
                }
                
                result = upload_image_to_cloudinary(mock_image_data, location.imageId, metadata)
                
                if result["success"]:
                    cloud_record = {
                        "imageId": location.imageId,
                        "imageName": location.imageName,
                        "cloudinaryUrl": result["url"],
                        "publicId": result["public_id"],
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "uploadedAt": datetime.now(),
                        "size": result.get("size", 0),
                        "format": result.get("format", "unknown")
                    }
                    cloud_storage_records.append(cloud_record)
                    
                    saved_images.append({
                        "imageId": location.imageId,
                        "imageName": location.imageName,
                        "cloudinaryUrl": result["url"],
                        "publicId": result["public_id"],
                        "status": "uploaded"
                    })
                else:
                    failed_images.append({
                        "imageId": location.imageId,
                        "imageName": location.imageName,
                        "error": result["error"],
                        "status": "failed"
                    })
                    
            except Exception as e:
                failed_images.append({
                    "imageId": location.imageId,
                    "imageName": location.imageName,
                    "error": str(e),
                    "status": "failed"
                })
        
        return BulkSaveResponse(
            success=len(failed_images) == 0,
            saved_count=len(saved_images),
            failed_count=len(failed_images),
            saved_images=saved_images,
            failed_images=failed_images,
            message=f"Bulk save completed: {len(saved_images)} saved, {len(failed_images)} failed"
        )
        
    except Exception as e:
        logger.error(f"Bulk save error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint from second file
@app.get("/cloud-storage")
async def get_cloud_storage_records():
    """Get all cloud storage records"""
    return {
        "records": cloud_storage_records,
        "total_uploaded": len(cloud_storage_records),
        "cloudinary_info": {
            "configured": bool(os.getenv("CLOUDINARY_CLOUD_NAME")),
            "cloud_name": os.getenv("CLOUDINARY_CLOUD_NAME", "not_configured")
        }
    }

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
    global processed_locations, decisions_log, cloud_storage_records
    processed_locations.clear()
    decisions_log.clear()
    cloud_storage_records.clear()
    
    return {"message": "All data cleared successfully"}

@app.get("/statistics")
async def get_statistics():
    """
    Get processing statistics.
    """
    total_locations = len(processed_locations)
    total_decisions = len(decisions_log)
    total_uploaded = len(cloud_storage_records)
    
    # Count decision types
    save_both = sum(1 for d in decisions_log if d["action"] == "save_both")
    keep_first = sum(1 for d in decisions_log if d["action"] == "keep_first_remove_second")
    keep_second = sum(1 for d in decisions_log if d["action"] == "remove_first_keep_second")
    
    return {
        "total_locations_processed": total_locations,
        "total_decisions_made": total_decisions,
        "total_images_uploaded": total_uploaded,
        "approved_images": len(get_approved_images()),
        "decisions_breakdown": {
            "save_both": save_both,
            "keep_first_remove_second": keep_first,
            "remove_first_keep_second": keep_second
        },
        "storage_info": {
            "cloudinary_configured": bool(os.getenv("CLOUDINARY_CLOUD_NAME")),
            "total_cloud_storage_used": sum(r.get("size", 0) for r in cloud_storage_records)
        }
    }

# New endpoint from second file
@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "model_loaded": mobilenet_model is not None,
        "tensorflow_version": tf.__version__,
        "cloudinary_configured": bool(os.getenv("CLOUDINARY_CLOUD_NAME")),
        "endpoints": [
            "/check-proximity", "/save-decision", "/classify-image", "/upload-image", 
            "/bulk-save", "/cloud-storage", "/statistics", "/model-status", "/reload-model"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🥭 Starting Mango Tree Location API with MobileNet and Cloudinary Storage...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🧠 Model Status: http://localhost:8000/model-status")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
