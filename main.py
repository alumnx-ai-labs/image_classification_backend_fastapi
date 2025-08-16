from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import random
import uuid
from datetime import datetime
import logging
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from bson import ObjectId

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mango Tree Location API", version="1.0.0")

#Setting up connection with the database
load_dotenv()
connection_string = os.getenv('MONGODB_URI')

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

class PlantLocation(BaseModel):
    cropType: str = "mango"  # Default to mango, but can be specified
    latitude: str
    longitude: str


class LocationRequest(BaseModel):
    locations: List[LocationData]

class FarmLocationInput(BaseModel):
    latitude: str
    longitude: str
    farmId: str  # Mandatory farm _id
    cropType: Optional[str] = "mango"  # Optional crop type, defaults to mango

class FarmLocationResponse(BaseModel):
    latitude: str
    longitude: str
    cropType: str
    farmId: str
    isDuplicate: bool
    saved: bool
    message: str

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

# Global variables
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

#Helper function for ObjectId serialization
def serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable format"""
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    if isinstance(doc, dict):
        return {key: serialize_doc(value) for key, value in doc.items()}
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc

# Retrieving Farm data from the database
def get_farm_data():
    client = None
    try:
        client = MongoClient(connection_string)
        client.admin.command('ping')

        try:
            db = client['agriculture']

            try:
                farm_collection = db['farm']

                try:
                    # Fetch all documents and serialize them
                    documents = list(farm_collection.find())
                    client.close()
                    return serialize_doc(documents)

                except Exception as e:
                    client.close()
                    return {"error": f"document access error: {str(e)}"}

            except Exception as e:
                client.close()
                return {"error": f"Collection access error: {str(e)}"}

        except Exception as e:
            client.close()
            return {"error": f"Database access error: {str(e)}"}        

    except Exception as e:
        return {"error": f"Database server connection error: {str(e)}"}

# Helper function to save single plant data to MongoDB
def save_plant_to_farm(latitude: str, longitude: str, farm_id: str, crop_type: str = "mango"):
    """Save single plant to existing farm after checking for duplicates"""
    client = None
    try:
        client = MongoClient(connection_string)
        client.admin.command('ping')
        
        db = client['agriculture']
        farm_collection = db['farm']
        
        # Convert farmId string to ObjectId
        from bson import ObjectId
        farm_object_id = ObjectId(farm_id)
        
        # Check if farm exists
        existing_farm = farm_collection.find_one({"_id": farm_object_id})
        
        if not existing_farm:
            client.close()
            return {
                "success": False,
                "isDuplicate": False,
                "message": f"Farm with ID {farm_id} not found"
            }
        
        # Check for duplicate coordinates in existing plants
        existing_plants = existing_farm.get("plants", [])
        for plant in existing_plants:
            if plant.get("latitude") == latitude and plant.get("longitude") == longitude:
                client.close()
                return {
                    "success": False,
                    "isDuplicate": True,
                    "message": "Plant with same coordinates already exists in this farm"
                }
        
        # No duplicate found, add new plant
        new_plant = {
            "cropType": crop_type,
            "latitude": latitude,
            "longitude": longitude
        }
        
        # Add plant to the farm's plants array
        result = farm_collection.update_one(
            {"_id": farm_object_id},
            {"$push": {"plants": new_plant}}
        )
        
        client.close()
        
        if result.modified_count > 0:
            return {
                "success": True,
                "isDuplicate": False,
                "message": f"Plant added successfully to farm {farm_id}"
            }
        else:
            return {
                "success": False,
                "isDuplicate": False,
                "message": "Failed to add plant to farm"
            }
        
    except Exception as e:
        if client:
            client.close()
        return {
            "success": False,
            "isDuplicate": False,
            "message": f"Database error: {str(e)}"
        }




@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Mango Tree Location API is running", 
        "timestamp": datetime.now()
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

@app.post("/save-farm-data", response_model=FarmLocationResponse)
async def save_farm_data(location: FarmLocationInput):
    try:
        logger.info(f"Processing plant data for farm {location.farmId}")
        
        # Validate farmId format (should be a valid ObjectId)
        try:
            from bson import ObjectId
            ObjectId(location.farmId)  # This will raise an exception if invalid
        except Exception as e:
            logger.error(f"Invalid farmId {location.farmId}: {str(e)}")
            return FarmLocationResponse(
                latitude=location.latitude,
                longitude=location.longitude,
                cropType=location.cropType or "mango",
                farmId=location.farmId,
                isDuplicate=False,
                saved=False,
                message=f"Invalid farmId format: {str(e)}"
            )
        
        # Save to database
        save_result = save_plant_to_farm(
            location.latitude, 
            location.longitude, 
            location.farmId,
            location.cropType or "mango"
        )
        
        if save_result["success"]:
            logger.info(f"Successfully saved plant to farm {location.farmId}")
        else:
            logger.warning(f"Failed to save plant to farm {location.farmId}: {save_result['message']}")
        
        # Create response
        return FarmLocationResponse(
            latitude=location.latitude,
            longitude=location.longitude,
            cropType=location.cropType or "mango",
            farmId=location.farmId,
            isDuplicate=save_result["isDuplicate"],
            saved=save_result["success"],
            message=save_result["message"]
        )
        
    except Exception as e:
        logger.error(f"Error processing plant data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing plant data: {str(e)}")



@app.get("/dashboard")
async def dashboard():
    """
    Return Data to Dispay on Dashboard.
    """

    farm_data = get_farm_data()

    if farm_data:
        return (farm_data)
    else:
        return("could not get farm data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")