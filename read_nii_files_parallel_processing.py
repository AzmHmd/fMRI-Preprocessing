# This code loads the nii files (in parallel) and saves them in to types: 1- all 3 views in one figure; 2) all individual slices in specified folders.
# Written by Azam Hamidinekoo, Aberystwyth University, 2019
#-------------------------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

from multiprocessing import Process, Queue
import scipy


def show_slices(slices):
	""" Function to display row of image slices """
	fig, axes = plt.subplots(1, len(slices))
	for i, slice in enumerate(slices):
		axes[i].imshow(slice.T, cmap="gray", origin="lower")
		axes[i].axis('off')



def saveAllViews(q, name, path):
	example_filename = os.path.join(path, name)
	img = nib.load(example_filename)
	img_data = img.get_fdata()

	print(' The shape of the 3D image is:')
	print(img_data.shape)

	#print(img.affine.shape)
	#print(img.header)

	# Save all views in one figure
	a = np.array(img_data.shape)
	for i in range(0, a.min()):
		slice_00 = img_data[i, :, :]
		slice_11 = img_data[:, i, :]
		slice_22 = img_data[:, :, i]
		slices_together = [slice_00, slice_11, slice_22]
		show_slices(slices_together)
		plt.suptitle(name)
		# Save the figures with all views
		folderName = '3Views_' + name.replace('.nii.gz','')
		if not os.path.exists(path+'/'+folderName):
			os.mkdir(path+'/'+folderName)
    	plt.savefig(path+'/'+folderName+'/'+str(i)+'.png')
    	plt.close('all')


def main():
	root = "/media/azh2/Seagate Backup Plus Drive/Azam/fMARI_data/Bangor/"
	for path, subdirs, files in os.walk(root):
	    for name in files:
	        print os.path.join(path, name)

	        if name.find('nii.gz') != -1 and name.find('rest.nii.gz') == -1:

	        	# save via parallel processing
		        q = Queue()
    			all_views = Process(target=saveAllViews, args=(q, name, path))
    			all_views.start()

		      
		        # Save the slices from different views	        

		        for i0 in range(0, img_data.shape[0]):
		        	slice_0 = img_data[i0, :, :]

		        	# save the slice0
		        	folderSlices_0 = 'view0_' + name.replace('.nii.gz','')
		        	if not os.path.exists(path+'/'+folderSlices_0):
		        		os.mkdir(path+'/'+folderSlices_0)
		        	plt.imsave(path+'/'+folderSlices_0+'/'+str(i0)+'.png', slice_0, cmap='gray', 
		        		vmin=slice_0.min(), vmax=slice_0.max())


		        for i1 in range(0, img_data.shape[1]):
		        	slice_1 = img_data[:, i1, :]

		        	# save the slice1
		        	folderSlices_1 = 'view1_' + name.replace('.nii.gz','')
		        	if not os.path.exists(path+'/'+folderSlices_1):
		        		os.mkdir(path+'/'+folderSlices_1)
		        	plt.imsave(path+'/'+folderSlices_1+'/'+str(i1)+'.png', slice_1, cmap='gray',
		        		vmin=slice_1.min(), vmax=slice_1.max())

		        for i2 in range(0, img_data.shape[2]):
		        	slice_2 = img_data[:, :, i2]

		        	# save the slice2
		        	folderSlices_2 = 'view2_' + name.replace('.nii.gz','')
		        	if not os.path.exists(path+'/'+folderSlices_2):
		        		os.mkdir(path+'/'+folderSlices_2)
		        	plt.imsave(path+'/'+folderSlices_2+'/'+str(i2)+'.png', slice_2, cmap='gray', 
		        		vmin=slice_2.min(), vmax=slice_2.max())


	plt.show()
	#plt.pause(.1)

if __name__ == '__main__':
	main()
		




