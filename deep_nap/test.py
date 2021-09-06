# from napari.types import LabelsData, ImageData
# from magicgui import magicgui
# from magicgui import magic_factory
# import napari
# import numpy as np
# from napari.layers import Image
# import json
# import requests

# url = 'http://127.0.0.1'
# port = "8000"

# # get model list
# model_list = requests.get(url + ':' + port + '/available_models')        
# model_list = json.loads(model_list.text)
# model_list = model_list['model_names']


# ## Load selected model
# @magicgui(call_button="Load", model_name={"choices": model_list})
# def load_model(model_name = model_list[0]):        
#     data = json.dumps({"model_name": model_name})    
#     headers = {"content-type": "application/json"}
#     json_response = requests.post(url + ':' + port + '/load_model', data=data, headers=headers, timeout=1800)     
#     print(json_response)
#     return data


# viewer = napari.Viewer()#view_image(np.random.rand(64, 64), name="My Image")
# viewer.window.add_function_widget(load_model)

# napari.run()
# #my_widget()
# #load_model.show(run=True)
import napari
import numpy as np
from napari.layers import Image
from magicgui import magicgui

@magicgui(image={'label': 'Pick an Image'})
def my_widget(image: Image):
    ...

viewer = napari.view_image(np.random.rand(64, 64), name="My Image")
viewer.window.add_dock_widget(my_widget)