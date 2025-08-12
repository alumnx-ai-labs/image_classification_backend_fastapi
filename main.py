from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import random
import uuid
from datetime import datetime
import math

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

# In-memory storage (replace with actual database in production)
processed_locations = []
decisions_log = []

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the distance between two GPS coordinates using Haversine formula.
    Returns distance in meters.
    """
    # TODO: Implement actual distance calculation logic
    # For now, return a mock distance for testing
    return random.uniform(0.5, 2.0)

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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Mango Tree Location API is running", "timestamp": datetime.now()}

@app.post("/check-proximity", response_model=ProximityResponse)
async def check_proximity(request: LocationRequest):
    """
    Receive locations and check if any two mango tree images are less than 1m apart.
    Returns similar image pairs/groups.
    """
    try:
        # Store locations globally
        global processed_locations
        processed_locations.extend(request.locations)
        
        # Find nearby pairs
        similar_pairs = find_nearby_pairs(request.locations)
        
        # TODO: Replace this with actual logic
        # Current implementation returns 2 random pairs for testing
        
        return ProximityResponse(
            similar_pairs=similar_pairs,
            total_images=len(request.locations),
            pairs_found=len(similar_pairs)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing locations: {str(e)}")

@app.post("/save-decision", response_model=DecisionResponse)
async def save_decision(request: DecisionRequest):
    """
    Save user's decision about duplicate image pairs to database.
    """
    try:
        # TODO: Implement actual database saving logic
        # For now, just log the decision
        
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
        raise HTTPException(status_code=500, detail=f"Error saving decision: {str(e)}")

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

# Additional endpoints for production use

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
    uvicorn.run(app, host="0.0.0.0", port=8000)