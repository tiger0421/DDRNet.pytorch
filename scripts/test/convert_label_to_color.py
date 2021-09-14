import cv2
import numpy as np
import time

color_map = {
    "Animal"              : (64, 128, 64  ), 
    "Archway"             : (192, 0, 128  ), 
    "Bicyclist"           : (0, 128, 192  ), 
    "Bridge"              : (0, 128, 64   ), 
    "Building"            : (128, 0, 0    ), 
    "Car"                 : (64, 0, 128   ), 
    "CartLuggagePram"     : (64, 0, 192   ), 
    "Child"               : (192, 128, 64 ), 
    "Column_Pole"         : (192, 192, 128), 
    "Fence"               : (64, 64, 128  ), 
    "LaneMkgsDriv"        : (128, 0, 192  ), 
    "LaneMkgsNonDriv"     : (192, 0, 64   ), 
    "Misc_Text"           : (128, 128, 64 ), 
    "MotorcycleScooter"   : (192, 0, 192  ), 
    "OtherMoving"         : (128, 64, 64  ), 
    "ParkingBlock"        : (64, 192, 128 ), 
    "Pedestrian"          : (64, 64, 0    ), 
    "Road"                : (128, 64, 128 ), 
    "RoadShoulder"        : (128, 128, 192), 
    "Sidewalk"            : (0, 0, 192    ), 
    "SignSymbol"          : (192, 128, 128), 
    "Sky"                 : (128, 128, 128), 
    "SUVPickupTruck"      : (64, 128, 192 ), 
    "TrafficCone"         : (0, 0, 64     ), 
    "TrafficLight"        : (0, 64, 64    ), 
    "Train"               : (192, 64, 128 ), 
    "Tree"                : (128, 128, 0  ), 
    "Truck_Bus"           : (192, 128, 192), 
    "Tunnel"              : (64, 0, 64    ), 
    "VegetationMisc"      : (192, 192, 0  ), 
    "Void"                : (0, 0, 0      ), 
    "Wall"                : (64, 192, 0   ), 
}

img = cv2.imread('result_label.png', cv2.IMREAD_GRAYSCALE)
result = np.zeros((img.shape[0], img.shape[1], 3))

start = time.time()
for i, v in enumerate(color_map.values()):
    result[img == i] = v
    if np.any(img==i):
        print(i, list(color_map.items())[i])

print(time.time()-start)
result = cv2.cvtColor(np.array(result, dtype=np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite("result_color.png", result)
