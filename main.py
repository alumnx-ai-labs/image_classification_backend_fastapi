from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import random
import uuid
from datetime import datetime
import math
import logging
import base64
import io
import os
from PIL import Image
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mango Tree Location API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Cloudinary (set these environment variables)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "your_cloud_name"),
    api_key=os.getenv("CLOUDINARY_API_KEY", "your_api_key"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "your_api_secret")
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
    action: str
    imageId1: str
    imageId2: str

class DecisionResponse(BaseModel):
    success: bool
    message: str
    saved_to_database: bool

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

# Global storage
processed_locations = []
decisions_log = []
cloud_storage_records = []

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
        "message": "Mango Tree Location API with Cloudinary Storage",
        "version": "2.0.0",
        "timestamp": datetime.now(),
        "cloudinary_configured": bool(os.getenv("CLOUDINARY_CLOUD_NAME"))
    }

@app.post("/check-proximity", response_model=ProximityResponse)
async def check_proximity(request: LocationRequest):
    """Check proximity between locations"""
    try:
        global processed_locations
        processed_locations.extend(request.locations)
        similar_pairs = find_nearby_pairs(request.locations)
        
        logger.info(f"Found {len(similar_pairs)} pairs from {len(request.locations)} locations")
        
        return ProximityResponse(
            similar_pairs=similar_pairs,
            total_images=len(request.locations),
            pairs_found=len(similar_pairs)
        )
    except Exception as e:
        logger.error(f"Proximity check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-decision", response_model=DecisionResponse)
async def save_decision(request: DecisionRequest):
    """Save user decision about image pairs"""
    try:
        global decisions_log
        
        decision_record = {
            "id": str(uuid.uuid4()),
            "pairId": request.pairId,
            "action": request.action,
            "imageId1": request.imageId1,
            "imageId2": request.imageId2,
            "timestamp": datetime.now(),
            "saved_to_db": True
        }
        
        decisions_log.append(decision_record)
        logger.info(f"Decision saved: {request.action} for pair {request.pairId}")
        
        return DecisionResponse(
            success=True,
            message=f"Decision '{request.action}' saved successfully",
            saved_to_database=True
        )
    except Exception as e:
        logger.error(f"Save decision error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/statistics")
async def get_statistics():
    """Get processing statistics"""
    total_locations = len(processed_locations)
    total_decisions = len(decisions_log)
    total_uploaded = len(cloud_storage_records)
    
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

@app.delete("/clear-data")
async def clear_data():
    """Clear all stored data"""
    global processed_locations, decisions_log, cloud_storage_records
    processed_locations.clear()
    decisions_log.clear()
    cloud_storage_records.clear()
    
    return {"message": "All data cleared successfully"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "cloudinary_configured": bool(os.getenv("CLOUDINARY_CLOUD_NAME")),
        "endpoints": [
            "/check-proximity", "/save-decision", "/upload-image", 
            "/bulk-save", "/cloud-storage", "/statistics"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🥭 Starting Mango Tree Location API with Cloudinary Storage...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("\n📝 Setup Instructions:")
    print("1. Sign up at https://cloudinary.com (free, no credit card)")
    print("2. Set environment variables:")
    print("   export CLOUDINARY_CLOUD_NAME='your_cloud_name'")
    print("   export CLOUDINARY_API_KEY='your_api_key'")
    print("   export CLOUDINARY_API_SECRET='your_api_secret'")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
