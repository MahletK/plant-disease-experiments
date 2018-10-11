
# coding: utf-8

# In[1]:


import os
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import subprocess
from keras.applications.inception_v3 import preprocess_input
import argparse


# In[2]:


parser = argparse.ArgumentParser()

# img_path = 'abc.jpg'
# img_seg_path = 'abc_marked.jpg'

target_size = (64, 64)


# In[3]:


def predict(img_path):
    image_name,extension=os.path.splitext(img_path)
    new_image = image_name+"_marked"+extension
#     print(new_image)
    result = subprocess.check_output(['python', "leaf-image-segmentation/segment.py", "-s", img_path])
    model_path = os.path.join('Plant_Disease_Detection_Benchmark_models/Models', 'VGG_scratch_94.h5')
    model = load_model(model_path)
    img = Image.open(img_path)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    label = np.argmax(preds)
    return label


# In[4]:


# if __name__ == "__main__":
#     parser.add_argument("--image")
#     args = parser.parse_args()
#     predict(args.image)
a = predict('apple_scab_marked.JPG')
print(a)


# In[7]:


x = [0] * 38
print(x)


# In[10]:


import os
i = 0
img_path = 'Apple___Cedar_apple_rust/'
for f in os.listdir('Apple___Cedar_apple_rust/'):
#     print(f)
    result = predict(img_path + f)
    print(result)
    x[result] += 1
    print(x[result])
    print(x)

# y = 0
# for i in range(5):
#     y += 1
#     print(y)
# In[ ]:


import os
i = 0
img_path = 'Apple___Cedar_apple_rust/'
for f in os.listdir('Apple___Cedar_apple_rust/'):
#     print(f)
    result = predict(img_path + f)
    print(result)
    x[result] += 1
    print(x[result])
    print(x)

