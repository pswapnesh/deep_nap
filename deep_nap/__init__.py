try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

## import required functions
import os
from napari_plugin_engine import napari_hook_implementation
from magicgui import magic_factory
from magicgui.tqdm import trange
import numpy as np
import requests 
import json


'''
Required endpoints:
"/predict?data=" for predictions
'/available_models' for available models list
'/load_model' to load specified model

defaults:
url = 'http://127.0.0.1'
port = "8000"

'''
# API parameters initialization
url = 'http://127.0.0.1'
port = "8000"
predict_point = "/predict?data="
meta_point = '/available_models'
API_ENDPOINT = url + ":" + port + predict_point

# get model list
model_list = requests.get(url + ':' + port + meta_point)        
model_list = json.loads(model_list.text)
model_list = model_list['model_names']

threshold = 0.92

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
        predictions = np.zeros_like(image)[:,:,np.newaxis]
    return np.array(predictions)[:,:,0]


## API params
@magic_factory(call_button="set")
def settings(api_endpoint = 'http://127.0.0.1', portnum = "8000"):        
    url = api_endpoint
    port = portnum    
    # get model list
    model_list = requests.get(url + ':' + port + meta_point)        
    model_list = json.loads(model_list.text)
    model_list = model_list['model_names']


## Load selected model
@magic_factory(call_button="Load", model_name={"choices": model_list})
def load_model(model_name = model_list[0]):        
    data = json.dumps({"model_name": model_name})    
    headers = {"content-type": "application/json"}
    json_response = requests.post(url + ':' + port + '/load_model', data=data, headers=headers, timeout=1800) 
    


## MagicGui widget for single image segmentation
@magic_factory(call_button="Segment")
def segment(data: 'napari.types.ImageData') -> 'napari.types.ImageData':        
    if len(data.shape) == 2:        
        pred = api_prediction(API_ENDPOINT,data)
    else:
        pred = np.array([api_prediction(API_ENDPOINT,d) for d in trange(data)])
    return pred


## MagicGui widget for single image segmentation
@magic_factory(auto_call=True,threshold = {"widget_type": "Slider", "max": 254}) #, "tracking": False
def post_process(data: 'napari.types.ImageData', threshold = 240) -> 'napari.types.LabelsData':            
    return (data > threshold)

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [settings,load_model,segment,post_process]


# @napari_hook_implementation
# def napari_experimental_provide_dock_widget():
#     return segment


