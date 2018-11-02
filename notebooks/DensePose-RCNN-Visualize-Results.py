#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np

im  = cv2.imread('../DensePoseData/demo_data/demo_im.jpg')
IUV = cv2.imread('../DensePoseData/infer_out/demo_im_IUV.png')
INDS = cv2.imread('../DensePoseData/infer_out/demo_im_INDS.png',  0)


# Let's visualize the I, U and V images.

# In[79]:


fig = plt.figure(figsize=[15,15])
plt.imshow(   np.hstack((IUV[:,:,0]/24. ,IUV[:,:,1]/256. ,IUV[:,:,2]/256.))  )
plt.title('I, U and V images.')
plt.axis('off') ; plt.show()


# Let's visualize the isocontours of the UV fields.

# In[80]:


fig = plt.figure(figsize=[12,12])
plt.imshow( im[:,:,::-1] )
plt.contour( IUV[:,:,1]/256.,10, linewidths = 1 )
plt.contour( IUV[:,:,2]/256.,10, linewidths = 1 )
plt.axis('off') ; plt.show()


# Let's visualize the human-body FG mask indices.

# In[81]:


fig = plt.figure(figsize=[12,12])
plt.imshow( im[:,:,::-1] )
plt.contour( INDS, linewidths = 4 )
plt.axis('off') ; plt.show()

