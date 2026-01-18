#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


get_ipython().system('pip install git+https://github.com/seppe-intelliprove/face-detection-onnx')


# In[ ]:


from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image

import PIL
from IPython.display import display

def detect_faces(image: PIL.Image):
    detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
    faces = detect_faces(image)
    print(f"Found {len(faces)} faces")
    return faces


def mark_faces(image_filename):
    """Mark all faces recognized in the image"""
    image = PIL.Image.open(image_filename)

    faces = detect_faces(image)

    # Draw faces
    render_data = detections_to_render_data(
        faces, bounds_color=Colors.GREEN, line_width=3
    )
    render_to_image(render_data, image)

    display(image)


# In[ ]:


get_ipython().system('wget https://upload.wikimedia.org/wikipedia/commons/3/3d/Apollo_11_Crew.jpg')
mark_faces("Apollo_11_Crew.jpg")


# In[ ]:


get_ipython().system('wget https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/Isabella_L%C3%B6vin_signing_climate_law_referral.jpg/1024px-Isabella_L%C3%B6vin_signing_climate_law_referral.jpg -O IL.jpg')
mark_faces("IL.jpg")


# In[ ]:


get_ipython().system('wget https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/20180610_FIFA_Friendly_Match_Austria_vs._Brazil_Miranda_850_0051.jpg/1024px-20180610_FIFA_Friendly_Match_Austria_vs._Brazil_Miranda_850_0051.jpg -O FIFA.jpg')
mark_faces("FIFA.jpg")


# In[ ]:




