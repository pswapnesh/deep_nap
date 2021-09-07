try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

## import required functions
import os
from napari_plugin_engine import napari_hook_implementation
from magicgui import magic_factory
from magicgui.tqdm import trange
from skimage.transform import rescale, resize
from scipy.ndimage import label
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

# global params
Segmentation = False

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
    

@magic_factory(auto_call=True,scale = {"widget_type": "FloatSlider", "min":0.5,"max": 3.0, "tracking": False})
def quickscale(data: 'napari.types.ImageData', scale = 1):    
    sr,sc = data.shape[-2],data.shape[-1]
    size = 256
    if scale==1:
        points = np.array([[int(sr/2), int(sc/2)]])
        points_layer = napari.viewer.add_points(points, size=30,name = 'ROI')
    
    points = napari.viewer.layer('ROI').data
    point = points[-1]
    r1 = sr if point[0]+size > sr else point[0]+size
    r0 = r1-size
    c1 = sc if point[1]+size > sc else point[1]+size
    c0 = c1 - size
    if scale > 1:
        im = rescale(data,scale)

    # lines = napari.viewer.layers['Shapes'].data
    # d = 0
    # for l in lines:
    #     d += l[-1]**2+l[-2]**2
    # d/=len(lines)
    # d = np.sqrt(d)



## MagicGui widget for single image segmentation
@magic_factory(call_button="Segment")
def segment(data: 'napari.types.ImageData',scale = 1.0) -> 'napari.types.ImageData':            
    sr,sc = data.shape[-2],data.shape[-1]
    rescaling = (lambda x: x ) if scale==1 else (lambda x: rescale(x,scale))
    resizing = (lambda x: x ) if scale==1 else (lambda x: resize(x,[sr,sc]))
    
    if len(data.shape) == 2:        
        image = rescaling(data,scale)
        pred = resizing(api_prediction(API_ENDPOINT,image))                    
    else:       
        pred = np.array([resizing(api_prediction(API_ENDPOINT,rescaling(d))) for d in trange(data)])    
    Segmentation = True
    return pred


## MagicGui widget for single image segmentation
@magic_factory(auto_call=True,threshold = {"widget_type": "Slider", "max": 254}) #, "tracking": False
def post_process(data: 'napari.types.ImageData', threshold = 240) -> 'napari.types.LabelsData':  
    if Segmentation:           
        labelled,count = label(data > threshold)
        return labelled

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [settings,load_model,segment,post_process]


# @napari_hook_implementation
# def napari_experimental_provide_dock_widget():
#     return segment


