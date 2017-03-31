# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:48:46 2017

@author: othmane
"""

import pickle as pkl
import numpy as np

def load_data(file_name):
	""" 
	Loads the data from the file_name
	@return : X
	We suppose that each letter is a series of lines, as if we're drawing the letter withouth picking up the pen and only using lines 
	Each element of X is an angle degree between two consecutive lines
	@return : Y 
	The label of each element of X (exemple : 'a', 'g',...)
	"""
	
	data = pkl.load(file(file_name,"rb"))
	X = np.array(data.get('letters'))
	Y = np.array(data.get('labels'))
	
	return X,Y