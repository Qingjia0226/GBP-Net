import numpy as np
import time

def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)

def crop_volume_init(data, sz, st=(0, 0, 0)):   
    st = np.array(st).astype(np.int32)
    zmax=data.shape[0]
    ymax=data.shape[1]
    xmax=data.shape[2]
    
    z1=int(st[0]-0.5*sz[0])
    z2=int(st[0]+0.5*sz[0])
    y1=int(st[1]-0.5*sz[1])
    y2=int(st[1]+0.5*sz[1])
    x1=int(st[2]-0.5*sz[2])
    x2=int(st[2]+0.5*sz[2])
    
    st[0]=z1
    if z1<0:
      st[0]=0
    if z2>=zmax:
      st[0]=zmax-sz[0]-5
      
    st[1]=y1
    if y1<0:
      st[1]=0
    if y2>=ymax:
      st[1]=ymax-sz[1]-5
      
    st[2]=x1
    if x1<0:
      st[2]=0
    if x2>=xmax:
      st[2]=xmax-sz[2]-5
    

    aa=crop_volume_init(data,sz,st)
    if aa.shape==tuple(sz):
         return aa
    else:
         print('!!!!!!!!!!!!!!!',aa.shape,tuple(sz))
         return aa
   
   
   
         
def crop_volume(data, sz, st=(0, 0, 0)):
    # must be (z, y, x) or (c, z, y, x) format
    assert data.ndim in [3, 4]
    st = np.array(st).astype(np.int32)

    if data.ndim == 3:
        return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]
    else: # crop spatial dimensions
        return data[:, st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]

def crop_volume_l(data, sz, p):
    data = np.pad(data, ((50, 50), (150, 150), (150, 150)), 'reflect')
    strat_z = p[0] + 50 - sz[0]
    strat_y = p[1] + 150 - sz[1]
    strat_x = p[2] + 150 - sz[2]
    end_z = strat_z + 3 * sz[0]
    end_y = strat_y + 3 * sz[1]
    end_x = strat_x + 3 * sz[2]
    #print(strat_z,end_z, strat_y,end_y, strat_x,end_x)
    ROI = data[strat_z:end_z, strat_y:end_y, strat_x:end_x]
    return ROI
    
