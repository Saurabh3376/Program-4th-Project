import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator,TransformerMixin

## this is specifically for columns we know are meant to be numeric or object and there is no data conversion but have to impute the missing values
class DataFrameImputer(BaseEstimator,TransformerMixin):
	def __init__(self):
		self.impute_dict = {}
		self.feature_names = []

	def fit(self,x,y=None): ## where we are learning
		self.feature_names = x.columns

		for col in x.columns:
			## col is now the key or the column name and we have to learn what value would we select to replace or fill
			if x[col].dtype =='O':
				self.impute_dict = 'missing'
			else:
				self.impute_dict = x[col].median()

		return self

	def transform(self,x,y=None): ## now we have median and the 
		## the values that we have learnt
		return x.fillna(impute_dict)

	def get_feature_names(self):
		return self.feature_names


## in custom fico we are not learning anything seedha perform the needed operation 
class custom_fico(BaseEstimator,TransformerMixin):
	def __init__(self):
		## as we don't need any attribute to be used later we want the column to be called fico later instead of fico range
		self.feature_names = ['FICO']

	def fit(self,x,y=None):
		return self

	def transform(self,x):
		k = x['FICO.Range'].str.split('-',expand=True).astype(float)
		fico = 0.5*(k[0]+k[1])
		## we need to return this in the form of a DataFrame
		return pd.DataFrame({'fico':fico})

	def get_feature_names(self):
		return self.feature_names

## we just want the relevant features here 
class VarSelect(TransformerMixin,BaseEstimator):
	def __init__(self,feature_names):
		self.feature_names = feature_names

	def fit(self,x,y=None):
		return self

	def transform(self,x):
		return x[self.feature_names]

	def get_feature_names(self):
		return self.feature_names


class string_clean(BaseEstimator,TransformerMixin):
	def __init__(self,replace_it,replace_with):
		self.replace_it = replace_it
		self.replace_with = replace_with
		self.feature_names = []

	def fit(self,x,y=None):
		self.feature_names = x.columns
		return self

	def transform(self,x):
		for col in x.columns:
			x[col] = x[col].str.replace(replace_it,replace_with)
		return x 

	def get_feature_names(self):
		return self.feature_names


class convert_to_numeric(BaseEstimator,TransformerMixin):
	def __init__(self):
		self.feature_names = []

	def fit(self,x,y=None):
		self.feature_names = x.columns
		return self

	def transform(self,x):
		for col in x.columns:
			x[col] = pd.to_numeric(x[col],errors='coerce')
		retun x 

	def get_feature_names(self):
		return self.feature_names


class get_Dummies_pipe(BaseEstimator,TransformerMixin):
	## by defauly I am setting the frequency cutoff to be zero
	def __init__(self,freq_cutoff=0):
		self.freq_cutoff = freq_cutoff
		self.var_cat_dict = {} ## this will store the cateogories that do qualify in a dict
		self.feature_names = []

	def fit(self,x,y=None):

	for col in x.columns:
			## we have to check for those where there are no categories in the column that have frequency less than cutoff
			## in that case we have to just exclude the final column 
			k = x[col].value_counts()
			if (k<self.freq_cutoff).sum() == 0:
	## k.index is a list and now we have to exclude the last one hence the categories will be all except last we are learning not transforming yet
				cats = k.index[:-1]

			else:
				cats = k.index[k>self.freq_cutoff]
			## for each column there will be categories
			self.var_cat_dict[col] = cats
	## now what will be the feature names ?
	## feature names will be the dummies like state_ca and state_ny

	for col in self.var_cat_dict.keys():
		for cat in self.var_cat_dict[col]:
			self.feature_names.append(col+"_"+cat)



	return self


	## now we will impute the dictionary with these categories

	def transform(self,x,y=None):
		dummy_data = x.copy()
		for col in self.var_cat_dict.keys():
			for cat in self.var_cat_dict(col):
				name = col+"_"+cat
				dummy_data[name] = (dummy_data[col]==cat).astype(int) 
			del dummy_data[col]
		return dummy_data

	def get_feature_names(self):
		return self.feature_names
	





