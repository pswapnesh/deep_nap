try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

## import required functions
import os
from napari_plugin_engine import napari_hook_implementation
from magicgui import magic_factory
import numpy as np
import requests 
import json

API_ENDPOINT = "http://127.0.0.1:80/predict?data="
threshold = 0.65
# api_prediction
def api_prediction(API_ENDPOINT,image,invert = False):
    data = json.dumps({"light_background": invert,"instances": image.tolist()})    
    headers = {"content-type": "application/json"}
    json_response = requests.post(API_ENDPOINT, data=data, headers=headers, timeout=1800)        
    try:
        predictions = json.loads(json_response.text)
        predictions = predictions['prediction']        
    except:
        print('error')
        print(json_response)
        predictions = np.zeros_like(image)
    return np.array(predictions)

## MagicGui widget for single image segmentation
@magic_factory(auto_call=True)
def segment(data: 'napari.types.ImageData', 
) -> 'napari.types.LabelsData':    
    if len(data.shape) == 2:
        
        pred = api_prediction(API_ENDPOINT,data)
    else:
        pred = np.array([api_prediction(API_ENDPOINT,d) for d in data])
    return pred > threshold

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return segment

