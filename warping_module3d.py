import tensorflow as tf
import numpy as np
import time




#for main body mask and head mask (rotation of plane)
#mask input shape=(3,:)
def rotation_estimation_3joint(joint1i,joint2i,joint3i,joint1f,joint2f,joint3f,mask):
  midi=(joint1i+joint2i)/2
  midf=(joint1f+joint2f)/2
  

  ai=joint1i-midi
  af=joint1f-midf
  bi=joint2i-midi
  bf=joint2f-midf
  ci=joint3i-midi
  cf=joint3f-midf
  
  
  scle=tf.norm(bi-ai)/tf.norm(bf-af)

  Mi=np.column_stack((ai*scle,bi*scle,ci*scle))
  Mf=np.column_stack((af,bf,cf))

  rotation_mat=tf.linalg.matmul((Mf),tf.linalg.pinv(Mi)) #s*R(ji-mi)=jf-mf
  midi=np.reshape(midi,(3,-1))

  mask=np.reshape(mask,(3,-1))
  
 
  midf=np.reshape(midf,(3,-1))
  maskf=np.matmul(rotation_mat,mask-midi)*scle+midf
  
  return maskf #mask's final coordinate

def rotation_estimation(joint1i,joint2i,joint1f,joint2f,mask):
  
  a=joint1i-joint2i
  b=joint1f-joint2f
  
  
  scle=np.linalg.norm([b[0],b[1]])/np.linalg.norm([a[0],a[1]])
  
  angle=tf.math.atan(b[1]/b[0])-tf.math.atan(a[1]/a[0])
  si=tf.math.sin(angle)
  co=tf.math.cos(angle)
  
  rotation_mat=np.array([[co,-si],[si,co]])
  
  mxy=np.array([mask[0,:],mask[1,:]])
  j2i=np.array([[joint2i[0]],[joint2i[1]]])
  j2f=np.array([[joint2f[0]],[joint2f[1]]])
  
  xymask=np.matmul(rotation_mat,mxy-j2i)*scle+j2f
  
  
  zmask=((xymask[0,:]-joint2f[0])*b[2]/b[0])+joint2f[2]
  
  zmask=np.reshape(zmask,(1,-1))
  maskf=np.concatenate((xymask,zmask),axis=0)
  
  return maskf

def warpingModule(mask,transform,joint):
    warped_mask=[]

    warped_mask.append(rotation_estimation(joint['lsho'],joint['lelb'],transform['lsho'],transform['lelb'],mask[0]))
    warped_mask.append(rotation_estimation(joint['rsho'],joint['relb'],transform['rsho'],transform['relb'],mask[1]))
    warped_mask.append(rotation_estimation(joint['lelb'],joint['lwri'],transform['lelb'],transform['lwri'],mask[2]))
    warped_mask.append(rotation_estimation(joint['relb'],joint['rwri'],transform['relb'],transform['rwri'],mask[3]))
    warped_mask.append(rotation_estimation(joint['lhip'],joint['lkne'],transform['lhip'],transform['lkne'],mask[4]))
    warped_mask.append(rotation_estimation(joint['rhip'],joint['rkne'],transform['rhip'],transform['rkne'],mask[5]))
    warped_mask.append(rotation_estimation(joint['lkne'],joint['lank'],transform['lkne'],transform['lank'],mask[6]))
    warped_mask.append(rotation_estimation(joint['rkne'],joint['rank'],transform['rkne'],transform['rank'],mask[7]))
    warped_mask.append(rotation_estimation(joint['lear'],joint['rear'],joint['reye'],transform['lear'],transform['rear'],transform['reye'],mask[8]))
    warped_mask.append(rotation_estimation(joint['neck'],joint['pelv'],joint['rsho'],transform['neck'],transform['pelv'],transform['rsho'],mask[9]))

    return warped_mask


#Testing:
#dt = time.time()

#ji=np.array([3,-4,3])
#jf=np.array([4.99,6.26,11.9])
#j2i=np.array([-5,-7,-4])
#j2f=np.array([3,-2,-6])
#mask=np.array([[2.29,5.38,3.11,3],[0.07,2.43,3.92,2.5],[6.79,8.66,10,7.5]])
#c=rotation_estimation(ji,jf,j2i,j2f,mask)
#print(c)
#df = time.time()

#print('1 mask coordinate is generated in:',(df-dt)/4,'ms')
