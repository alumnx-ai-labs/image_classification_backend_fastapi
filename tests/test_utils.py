from main import calculate_distance

def test_calculate_distance_zero():
    lat = 56.7
    long = 10.3
    distance = calculate_distance(lat, long, lat, long)
    assert distance == 0, f"Expected 0 got {distance}"

def test_calculate_distance_accuracy():
    lat1, long1, lat2, long2 = (28.6139, 77.2090, 19.0760, 72.8777)
    distance = calculate_distance(lat1, long1, lat2, long2)

    # The distance between Delhi & Mumbai nearly 1150 km
    assert 1140000 <= distance <= 1160000, f"Expected distance ~1150 km, got {distance}"
    
    lat1, long1, lat2, long2 = (51.5074, -0.1278, 48.8566, 2.3522)
    distance = calculate_distance(lat1, long1, lat2, long2) 

    assert 342000 <= distance <= 344000, f"Expected distance ~343 km, got {distance}"