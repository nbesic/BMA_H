#	"Remote sensing-based forest canopy height mapping:
# 	some models are useful, but might they be even more so
# 	if combined?"
#
#	Complementary code 2 (Fig. 6 and A1)
#	Please see Main code (Fig. 3, 4, 5, 7 and B1)
#	Please see complementary code 1 (Fig. 2)
#
#	Fig. 1 and A2 represent respectively a diagram and 
#	an illustration and are established using TikZ and QGIS 
#	and are therefore not reproductable with the python code

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import lognorm
import seaborn as sns
import matplotlib.colors as colors
import copy
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy.ma as ma
import random
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import gridspec
from matplotlib.colors import Normalize

def py_sweep(a,b):

		return a*b

rgb_misc = np.loadtxt("GMD_data_input/misc_div.txt") # downloadable at https://github.com/IPCC-WG1/colormaps
rgb_slev = np.loadtxt("GMD_data_input/slev_div.txt") # downloadable at https://github.com/IPCC-WG1/colormaps

#create the colormap
IPCC_misc = colors.LinearSegmentedColormap.from_list('colormap', rgb_misc)
IPCC_slev = colors.LinearSegmentedColormap.from_list('colormap', rgb_slev)
 
my_dpi = 500

fs = 10

nbnn = 50
 
## Data preparation 

NFI_ind = 4 # the type of the NFI plot height employed

if NFI_ind == 1:
	strh = "H_max" # maximum height without imputations
elif NFI_ind == 2:
	strh = "H_mean" # mean height without imputations
elif NFI_ind == 3:
	strh = "H_dom" # dominant height without imputations
elif NFI_ind == 4:
	strh = "Hi_max" # maximum height with imputations
	str_mat = "$H^i_{max}$"
elif NFI_ind == 5:
	strh = "Hi_mean" # mean height with imputations
	str_mat = "$H^i_{mean}$"
elif NFI_ind == 6:
	strh = "Hi_dom" # dominant height with imputations
	str_mat = "$H^i_{dom}$"
elif NFI_ind == 7:
	strh = "Hi_lor" # Loray's height with imputations
	str_mat = "$H^i_{lor}$"
	
# Reading the csv file contaning the NFI plot heights along with other variables, as well as
# the extractions from 5 diffent models (maps) at these plots.

# This file is not available for download, due to the statistical secret applied on the exact position of plots,
# but the confidential access could be provided for the editor and reviewers in order to enable peer review.

data = pd.read_csv("GMD_data_input/Input_data_table.csv")

fs1 = 11

ill_mmc = 0

if ill_mmc == 1:

	## Illustration - mutual models' comparisons (Fig. A1 in the manuscript)

	data_1 = data.copy()
	data_1["M_3"] = data_1["M_3"]/100
	data_1["M_5"] = data_1["M_5"]/100

	fig = plt.figure(figsize=(12, 12))

	ax1 = plt.subplot2grid((4, 4), (0, 0))

	im1 = sns.kdeplot(data = data_1, x = "M_2", y = "M_1", color = 'k', legend = False)
	
	ax1.plot([0, np.nanmax(data_1["M_2"])], [0, np.nanmax(data_1["M_1"])], color = "k", linestyle = 'dashed', linewidth = 1)

	ax1.set_xlabel("$M_2$",fontsize = fs1-1)
	ax1.set_ylabel("$M_1$",fontsize = fs1-1)

	a=ma.masked_invalid(data_1["M_2"])
	b=ma.masked_invalid(data_1["M_1"])

	msk = (~a.mask & ~b.mask)

	Corr = np.ma.corrcoef(a[msk], b[msk])

	ax1.text(0.5, 0.95, "Pearson correlation r = %.2f" % Corr[0,1], transform = ax1.transAxes, ha = "center", fontsize = fs-1)

	plt.xticks(fontsize = fs1-2)
	plt.yticks(fontsize = fs1-2)

	ax1.set_aspect('equal')

	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)

	ax1.spines['bottom'].set_color((0.48402921953094963, 0.4543329488658208, 0.7106651287966167))
	ax1.spines['left'].set_color((0.23192618223760095, 0.5456516724336793, 0.7626143790849673))
	ax1.xaxis.label.set_color((0.48402921953094963, 0.4543329488658208, 0.7106651287966167))
	ax1.yaxis.label.set_color((0.23192618223760095, 0.5456516724336793, 0.7626143790849673))
	ax1.tick_params(axis='x', colors=(0.48402921953094963, 0.4543329488658208, 0.7106651287966167))
	ax1.tick_params(axis='y', colors=(0.23192618223760095, 0.5456516724336793, 0.7626143790849673))

	ax2 = plt.subplot2grid((4, 4), (0, 1))

	im2 = sns.kdeplot(data = data_1, x = "M_3", y = "M_1", color = 'k', legend = False)
	
	ax2.plot([0, np.nanmax(data_1["M_3"])], [0, np.nanmax(data_1["M_1"])], color = "k", linestyle = 'dashed', linewidth = 1)

	ax2.set_xlabel("$M_3$",fontsize = fs1-1)
	ax2.set_ylabel("$M_1$",fontsize = fs1-1)

	a=ma.masked_invalid(data_1["M_3"])
	b=ma.masked_invalid(data_1["M_1"])

	msk = (~a.mask & ~b.mask)

	Corr = np.ma.corrcoef(a[msk], b[msk])

	ax2.text(0.5, 0.95, "Pearson correlation r = %.2f" % Corr[0,1], transform = ax2.transAxes, ha = "center", fontsize = fs-1)

	plt.xticks(fontsize = fs1-2)
	plt.yticks(fontsize = fs1-2)

	ax2.set_aspect('equal')

	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)

	ax2.spines['bottom'].set_color((0.907912341407151, 0.20284505959246443, 0.16032295271049596))
	ax2.spines['left'].set_color((0.23192618223760095, 0.5456516724336793, 0.7626143790849673))
	ax2.xaxis.label.set_color((0.907912341407151, 0.20284505959246443, 0.16032295271049596))
	ax2.yaxis.label.set_color((0.23192618223760095, 0.5456516724336793, 0.7626143790849673))
	ax2.tick_params(axis='x', colors=(0.907912341407151, 0.20284505959246443, 0.16032295271049596))
	ax2.tick_params(axis='y', colors=(0.23192618223760095, 0.5456516724336793, 0.7626143790849673))

	ax3 = plt.subplot2grid((4, 4), (0, 2))

	im3 = sns.kdeplot(data = data_1, x = "M_4", y = "M_1", color = 'k', legend = False)
	
	ax3.plot([0, np.nanmax(data_1["M_4"])], [0, np.nanmax(data_1["M_1"])], color = "k", linestyle = 'dashed', linewidth = 1)

	ax3.set_xlabel("$M_4$",fontsize = fs1-1)
	ax3.set_ylabel("$M_1$",fontsize = fs1-1)

	a=ma.masked_invalid(data_1["M_4"])
	b=ma.masked_invalid(data_1["M_1"])

	msk = (~a.mask & ~b.mask)

	Corr = np.ma.corrcoef(a[msk], b[msk])

	ax3.text(0.5, 0.95, "Pearson correlation r = %.2f" % Corr[0,1], transform = ax3.transAxes, ha = "center", fontsize = fs-1)

	plt.xticks(fontsize = fs1-2)
	plt.yticks(fontsize = fs1-2)

	ax3.set_aspect('equal')

	ax3.spines['top'].set_visible(False)
	ax3.spines['right'].set_visible(False)

	ax3.spines['bottom'].set_color((0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax3.spines['left'].set_color((0.23192618223760095, 0.5456516724336793, 0.7626143790849673))
	ax3.xaxis.label.set_color((0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax3.yaxis.label.set_color((0.23192618223760095, 0.5456516724336793, 0.7626143790849673))
	ax3.tick_params(axis='x', colors=(0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax3.tick_params(axis='y', colors=(0.23192618223760095, 0.5456516724336793, 0.7626143790849673))

	ax4 = plt.subplot2grid((4, 4), (0, 3))

	im4 = sns.kdeplot(data = data_1, x = "M_5", y = "M_1", color = 'k', legend = False)
	
	ax4.plot([0, np.nanmax(data_1["M_5"])], [0, np.nanmax(data_1["M_1"])], color = "k", linestyle = 'dashed', linewidth = 1)

	ax4.set_xlabel("$M_5$",fontsize = fs1-1)
	ax4.set_ylabel("$M_1$",fontsize = fs1-1)

	a=ma.masked_invalid(data_1["M_5"])
	b=ma.masked_invalid(data_1["M_1"])

	msk = (~a.mask & ~b.mask)

	Corr = np.ma.corrcoef(a[msk], b[msk])

	ax4.text(0.5, 0.95, "Pearson correlation r = %.2f" % Corr[0,1], transform = ax4.transAxes, ha = "center", fontsize = fs-1)

	plt.xticks(fontsize = fs1-2)
	plt.yticks(fontsize = fs1-2)

	ax4.set_aspect('equal')

	ax4.spines['top'].set_visible(False)
	ax4.spines['right'].set_visible(False)

	ax4.spines['bottom'].set_color((0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax4.spines['left'].set_color((0.23192618223760095, 0.5456516724336793, 0.7626143790849673))
	ax4.xaxis.label.set_color((0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax4.yaxis.label.set_color((0.23192618223760095, 0.5456516724336793, 0.7626143790849673))
	ax4.tick_params(axis='x', colors=(0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax4.tick_params(axis='y', colors=(0.23192618223760095, 0.5456516724336793, 0.7626143790849673))

	ax5 = plt.subplot2grid((4, 4), (1, 1))

	im5 = sns.kdeplot(data = data_1, x = "M_3", y = "M_2", color = 'k', legend = False)
	
	ax5.plot([0, np.nanmax(data_1["M_3"])], [0, np.nanmax(data_1["M_2"])], color = "k", linestyle = 'dashed', linewidth = 1)

	ax5.set_xlabel("$M_3$",fontsize = fs1-1)
	ax5.set_ylabel("$M_2$",fontsize = fs1-1)

	a=ma.masked_invalid(data_1["M_3"])
	b=ma.masked_invalid(data_1["M_2"])

	msk = (~a.mask & ~b.mask)

	Corr = np.ma.corrcoef(a[msk], b[msk])

	ax5.text(0.5, 0.95, "Pearson correlation r = %.2f" % Corr[0,1], transform = ax5.transAxes, ha = "center", fontsize = fs-1)

	plt.xticks(fontsize = fs1-2)
	plt.yticks(fontsize = fs1-2)

	ax5.set_aspect('equal')

	ax5.spines['top'].set_visible(False)
	ax5.spines['right'].set_visible(False)

	ax5.spines['bottom'].set_color((0.907912341407151, 0.20284505959246443, 0.16032295271049596))
	ax5.spines['left'].set_color((0.48402921953094963, 0.4543329488658208, 0.7106651287966167))
	ax5.xaxis.label.set_color((0.907912341407151, 0.20284505959246443, 0.16032295271049596))
	ax5.yaxis.label.set_color((0.48402921953094963, 0.4543329488658208, 0.7106651287966167))
	ax5.tick_params(axis='x', colors=(0.907912341407151, 0.20284505959246443, 0.16032295271049596))
	ax5.tick_params(axis='y', colors=(0.48402921953094963, 0.4543329488658208, 0.7106651287966167))

	ax6 = plt.subplot2grid((4, 4), (1, 2))

	im6 = sns.kdeplot(data = data_1, x = "M_4", y = "M_2", color = 'k', legend = False)
	
	ax6.plot([0, np.nanmax(data_1["M_4"])], [0, np.nanmax(data_1["M_2"])], color = "k", linestyle = 'dashed', linewidth = 1)

	ax6.set_xlabel("$M_4$",fontsize = fs1-1)
	ax6.set_ylabel("$M_2$",fontsize = fs1-1)

	a=ma.masked_invalid(data_1["M_4"])
	b=ma.masked_invalid(data_1["M_2"])

	msk = (~a.mask & ~b.mask)

	Corr = np.ma.corrcoef(a[msk], b[msk])

	ax6.text(0.5, 0.95, "Pearson correlation r = %.2f" % Corr[0,1], transform = ax6.transAxes, ha = "center", fontsize = fs-1)

	plt.xticks(fontsize = fs1-2)
	plt.yticks(fontsize = fs1-2)

	ax6.set_aspect('equal')

	ax6.spines['top'].set_visible(False)
	ax6.spines['right'].set_visible(False)

	ax6.spines['bottom'].set_color((0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax6.spines['left'].set_color((0.48402921953094963, 0.4543329488658208, 0.7106651287966167))
	ax6.xaxis.label.set_color((0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax6.yaxis.label.set_color((0.48402921953094963, 0.4543329488658208, 0.7106651287966167))
	ax6.tick_params(axis='x', colors=(0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax6.tick_params(axis='y', colors=(0.48402921953094963, 0.4543329488658208, 0.7106651287966167))

	ax7 = plt.subplot2grid((4, 4), (1, 3))

	im7 = sns.kdeplot(data = data_1, x = "M_5", y = "M_2", color = 'k', legend = False)
	
	ax7.plot([0, np.nanmax(data_1["M_5"])], [0, np.nanmax(data_1["M_2"])], color = "k", linestyle = 'dashed', linewidth = 1)

	ax7.set_xlabel("$M_5$",fontsize = fs1-1)
	ax7.set_ylabel("$M_2$",fontsize = fs1-1)

	a=ma.masked_invalid(data_1["M_5"])
	b=ma.masked_invalid(data_1["M_2"])

	msk = (~a.mask & ~b.mask)

	Corr = np.ma.corrcoef(a[msk], b[msk])

	ax7.text(0.5, 0.95, "Pearson correlation r = %.2f" % Corr[0,1], transform = ax7.transAxes, ha = "center", fontsize = fs-1)

	plt.xticks(fontsize = fs1-2)
	plt.yticks(fontsize = fs1-2)

	ax7.set_aspect('equal')

	ax7.spines['top'].set_visible(False)
	ax7.spines['right'].set_visible(False)

	ax7.spines['bottom'].set_color((0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax7.spines['left'].set_color((0.48402921953094963, 0.4543329488658208, 0.7106651287966167))
	ax7.xaxis.label.set_color((0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax7.yaxis.label.set_color((0.48402921953094963, 0.4543329488658208, 0.7106651287966167))
	ax7.tick_params(axis='x', colors=(0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax7.tick_params(axis='y', colors=(0.48402921953094963, 0.4543329488658208, 0.7106651287966167))

	ax8 = plt.subplot2grid((4, 4), (2, 2))

	im8 = sns.kdeplot(data = data_1, x = "M_4", y = "M_3", color = 'k', legend = False)
	
	ax8.plot([0, np.nanmax(data_1["M_4"])], [0, np.nanmax(data_1["M_3"])], color = "k", linestyle = 'dashed', linewidth = 1)

	ax8.set_xlabel("$M_4$",fontsize = fs1-1)
	ax8.set_ylabel("$M_3$",fontsize = fs1-1)

	a=ma.masked_invalid(data_1["M_4"])
	b=ma.masked_invalid(data_1["M_3"])

	msk = (~a.mask & ~b.mask)

	Corr = np.ma.corrcoef(a[msk], b[msk])

	ax8.text(0.5, 0.95, "Pearson correlation r = %.2f" % Corr[0,1], transform = ax8.transAxes, ha = "center", fontsize = fs-1)

	plt.xticks(fontsize = fs1-2)
	plt.yticks(fontsize = fs1-2)

	ax8.set_aspect('equal')

	ax8.spines['top'].set_visible(False)
	ax8.spines['right'].set_visible(False)

	ax8.spines['bottom'].set_color((0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax8.spines['left'].set_color((0.907912341407151, 0.20284505959246443, 0.16032295271049596))
	ax8.xaxis.label.set_color((0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax8.yaxis.label.set_color((0.907912341407151, 0.20284505959246443, 0.16032295271049596))
	ax8.tick_params(axis='x', colors=(0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax8.tick_params(axis='y', colors=(0.907912341407151, 0.20284505959246443, 0.16032295271049596))

	ax9 = plt.subplot2grid((4, 4), (2, 3))

	im9 = sns.kdeplot(data = data_1, x = "M_5", y = "M_3", color = 'k', legend = False)
	
	ax9.plot([0, np.nanmax(data_1["M_5"])], [0, np.nanmax(data_1["M_3"])], color = "k", linestyle = 'dashed', linewidth = 1)

	ax9.set_xlabel("$M_5$",fontsize = fs1-1)
	ax9.set_ylabel("$M_3$",fontsize = fs1-1)

	a=ma.masked_invalid(data_1["M_5"])
	b=ma.masked_invalid(data_1["M_3"])

	msk = (~a.mask & ~b.mask)

	Corr = np.ma.corrcoef(a[msk], b[msk])

	ax9.text(0.5, 0.95, "Pearson correlation r = %.2f" % Corr[0,1], transform = ax9.transAxes, ha = "center", fontsize = fs-1)

	plt.xticks(fontsize = fs1-2)
	plt.yticks(fontsize = fs1-2)

	ax9.set_aspect('equal')

	ax9.spines['top'].set_visible(False)
	ax9.spines['right'].set_visible(False)

	ax9.spines['bottom'].set_color((0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax9.spines['left'].set_color((0.907912341407151, 0.20284505959246443, 0.16032295271049596))
	ax9.xaxis.label.set_color((0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax9.yaxis.label.set_color((0.907912341407151, 0.20284505959246443, 0.16032295271049596))
	ax9.tick_params(axis='x', colors=(0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax9.tick_params(axis='y', colors=(0.907912341407151, 0.20284505959246443, 0.16032295271049596))

	ax10 = plt.subplot2grid((4, 4), (3, 3))

	im10 = sns.kdeplot(data = data_1, x = "M_5", y = "M_4", color = 'k', legend = False)
	
	ax10.plot([0, np.nanmax(data_1["M_5"])], [0, np.nanmax(data_1["M_4"])], color = "k", linestyle = 'dashed', linewidth = 1)

	ax10.set_xlabel("$M_5$",fontsize = fs1-1)
	ax10.set_ylabel("$M_4$",fontsize = fs1-1)

	a=ma.masked_invalid(data_1["M_5"])
	b=ma.masked_invalid(data_1["M_4"])

	msk = (~a.mask & ~b.mask)

	Corr = np.ma.corrcoef(a[msk], b[msk])

	ax10.text(0.5, 0.95, "Pearson correlation r = %.2f" % Corr[0,1], transform = ax10.transAxes, ha = "center", fontsize = fs-1)

	plt.xticks(fontsize = fs1-2)
	plt.yticks(fontsize = fs1-2)

	ax10.set_aspect('equal')

	ax10.spines['top'].set_visible(False)
	ax10.spines['right'].set_visible(False)

	ax10.spines['bottom'].set_color((0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax10.spines['left'].set_color((0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax10.xaxis.label.set_color((0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax10.yaxis.label.set_color((0.9255363321799308, 0.3848673587081891, 0.05983852364475202))
	ax10.tick_params(axis='x', colors=(0.23044982698961936, 0.6445059592464436, 0.34514417531718566))
	ax10.tick_params(axis='y', colors=(0.9255363321799308, 0.3848673587081891, 0.05983852364475202))

	plt.savefig("GMD_illustrations/Models_comparisons.png", dpi=my_dpi)

valid_freq = 10 # k in k-fold cross validation

rd = 0 # 0 - to use and existing random sampling, any other number - to randomly resample

if rd == 0:

    ValMat = np.load("GMD_data_input/Fixed_sample_final_"+str(valid_freq)+".npy")

else:
	
	ValMat = np.zeros((valid_freq, np.int(np.floor(data.shape[0]/valid_freq))))
	
	train = np.arange(data.shape[0])
    
	for i in range(valid_freq):
		
		ValMat[i,:] = random.sample(list(train), np.int(np.floor(data.shape[0]/valid_freq)))
		
		train = np.setdiff1d(train, ValMat[i,:])
		
	np.save("GMD_data_input/Fixed_sample_final_"+str(valid_freq)+".npy", ValMat)

ValMat = ValMat.astype(int)

epsilon = 1e-6 # EM algorithm convergance parameters
maxiter = 1e5 # EM algorithm convergance parameters

pdf_ind = 1 # 1- normal, 2 - lognormal

if pdf_ind == 1:
	str_dist = "norm"
elif pdf_ind == 2:
	str_dist = "log_norm"

ZMat = np.ones((data.shape[0], 5, valid_freq))*-1

for valind in range(valid_freq):

	valid = ValMat[valind, :]
	
	train = np.setdiff1d(np.arange(data.shape[0]), valid)
	
	I=np.zeros((data.shape[0], 6))

	I[:,0]=data["M_1"].to_numpy() # Lang 2020 (m) - available for download at the link provided in the manuscript
	I[:,1]=data["M_2"].to_numpy() # Liu 2019 (m) - available for download at the link provided in the manuscript
	I[:,2]=data["M_3"].to_numpy()/100 # Morin 2020 (cm) - available for download at the link provided in the manuscript
	I[:,3]=data["M_4"].to_numpy() # Potapov 2019 (m) - available for download at the link provided in the manuscript
	I[:,4]=data["M_5"].to_numpy()/100 # Schwartz 2020 (cm) - available for download at the link provided in the manuscript

	I[:,5]=data[strh].to_numpy() # selected NFI height reference

	I = np.delete(I, valid, axis=0)

	### Bayesian model averaging

	# K - the total number of models (5 in our case)

	K = np.shape(I)[1] - 1 # 2nd dimension of input data - 1
	print(K)

	# N - the total number of observations (NFI plots)

	N = np.shape(I)[0] # 1st dimension of input data
	print(N)

	## Initialization of the Expectation - Maximization (EM) algorithm

	delta = np.inf
	iter = 0

	w = np.ones(K) / K # posterior model probability of the involved models

	w_1 = copy.copy(w)

	z = np.ones((N, K)) / K # posterior model probability of the involved models for the individual observations

	z_1 = copy.copy(z)

	sigma = np.ones((1, K)) # standard deviation of the involved models

	## Main loop of the Expectation - Maximization (EM) algorithm

	while (delta > epsilon) & (iter < maxiter):

		# E step

		if pdf_ind == 1:
			z_new = norm.pdf(np.tile(I[:, K][np.newaxis].T, [1, K]), I[:, range(K)], np.tile(sigma, [N, 1]))
		
		elif pdf_ind == 2:
			z_new = np.apply_along_axis(py_sweep, 1, lognorm.pdf(np.tile(I[:, K][np.newaxis].T, [1, K]), I[:, range(K)], np.tile(sigma, [N, 1])), w)

		z_new = z_new / np.tile(np.nansum(z_new, axis = 1)[np.newaxis].T,K)

		# M step
		
		w_new = np.nansum(z_new, axis = 0) / N

		sigma_new = np.sqrt(np.nansum(z_new*(np.tile(I[:, K][np.newaxis].T, [1, K])-I[:, range(K)])**2, axis = 0) / np.nansum(z_new, axis = 0))[np.newaxis]

		print(sigma_new)

		iter += 1
		print(iter)
		
		delta = np.linalg.norm(np.concatenate((w, sigma[0,:])) - np.concatenate((w_new, sigma_new[0,:])), ord=1)
		print(delta)
		
		z = z_new
		w = w_new
		sigma = sigma_new

	print(w_new)

	print(sigma_new)

	z_new_1 = copy.copy(z_new)

	z_new_1 = z_new_1 / np.nansum(z_new_1, axis = 1)[np.newaxis].T
	
	ZMat[train, :, valind] = z_new_1

ser_mask = gpd.read_file("GMD_data_input/ser_l93.shp") # available for download at the link https://inventaire-forestier.ign.fr/spip.php?article729 (in French) 

ser_mask = ser_mask.drop([149, 150])

SERcls = 	{'A11': 0, 'A12': 1, 'A13': 2, 'A21': 3, 'A22': 4, 'A30': 5,
			 'B10': 6, 'B21': 7, 'B22': 8, 'B23': 9, 'B31': 10, 'B32': 11, 'B33': 12, 'B41': 13, 'B42': 14, 'B43': 15, 'B44': 16, 'B51': 17, 'B52': 18, 'B53': 19,
			 'B61': 20, 'B62': 21, 'B70': 22, 'B81': 23, 'B82': 24, 'B91': 25, 'B92': 26,
			 'C11': 27, 'C12': 28, 'C20': 29, 'C30': 30, 'C41': 31, 'C42': 32, 'C51': 33, 'C52': 34,
			 'D11': 35, 'D12': 36,
			 'E10': 37, 'E20': 38,
			 'F11': 39, 'F12': 40, 'F13': 41, 'F14': 42, 'F15': 43, 'F21': 44, 'F22': 45, 'F23': 46, 'F30': 47, 'F40': 48, 'F51': 49, 'F52': 50,
			 'G11': 51, 'G12': 52, 'G13': 53, 'G21': 54, 'G22': 55, 'G23': 56, 'G30': 57, 'G41': 58, 'G42': 59, 'G50': 60, 'G60': 61, 'G70': 62, 'G80': 63, 'G90': 64,
			 'H10': 65, 'H21': 66, 'H22': 67, 'H30': 68, 'H41': 69, 'H42': 70,
			 'I11': 71, 'I12': 72, 'I13': 73, 'I21': 74, 'I22': 75,
			 'J10': 76, 'J21': 77, 'J22': 78, 'J23': 79, 'J24': 80, 'J30': 81, 'J40': 82,
			 'K11': 83, 'K12': 84, 'K13': 85}

spec=data['SER'].map(SERcls)
ser_orig=spec.to_numpy()

spec1 = ser_mask['codeser'].map(SERcls)
ser1 = spec1.to_numpy()

ser1 = ser1.astype(int)

SER_mat_orig = np.zeros((len(ser_orig),86))

for i in range(len(ser_orig)):
	for j in range(86):
		if ser_orig[i] == j:
			SER_mat_orig[i,j]+=1

SER_mat_nb_orig = np.sum(SER_mat_orig, axis=0)

ZMat[ZMat == -1] = np.nan

ZMat_CV = np.mean(np.nanstd(ZMat, axis = 2)/np.nanmean(ZMat, axis = 2), axis =1)*100

print(np.shape(ZMat_CV))

ZMat_CV_mat=np.zeros((np.shape(ZMat_CV)[0],86))

for i in range(np.shape(ZMat_CV)[0]):
	for j in range(86):
		if ser_orig[i] == j:
			ZMat_CV_mat[i,j] += ZMat_CV[i]

ZMat_CV_mat_mean = np.nansum(ZMat_CV_mat, axis=0) / (SER_mat_nb_orig - np.sum(np.isnan(ZMat_CV_mat) == 1, axis=0))

ser_mask["Weights_CV"] = ZMat_CV_mat_mean[ser1]

## Illustration - coefficients of variation (Fig. 6 in the manuscript)

fig = plt.figure(figsize=(10, 6))

ax1 = plt.subplot2grid((1, 1), (0, 0))

cmap1 = IPCC_misc

ser_mask.plot(ax = ax1, column = 'Weights_CV', cmap = cmap1, legend = True, legend_kwds={"shrink": 0.75, "format": "%.1f", "ticks": [0, 15, 30]}, vmin = 0, vmax = 30)

cbar = ax1.get_figure().axes[-1]
cbar.tick_params(labelsize = fs+4)

ax1.set_xticks([])
ax1.set_yticks([])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.set_title(str_mat+" - Coefficients of variation (%) \n (averaged by SER)", fontsize = fs+4)

plt.savefig("GMD_illustrations/CV_"+strh+"_"+str_dist+"_final_"+str(valid_freq)+"_alt.png", dpi=my_dpi)

fig = plt.figure(figsize=(10, 6))

ax2 = plt.subplot2grid((1, 1), (0, 0))

hist_data = sns.histplot(ZMat_CV, kde=True, stat="density", bins = 100, ax = ax2)

bin_centers = [patch.get_x() + patch.get_width() / 2 for patch in hist_data.patches]
heights = [patch.get_height() for patch in hist_data.patches]

# Normalize the x-values to [0,1] for colormap scaling
norm = Normalize(vmin=0, vmax=30)  # Values between 0 and 30
cmap = cmap1  # Choose a colormap, 'viridis' is one example

# Color each bar individually based on its x-value
for patch, x_value in zip(hist_data.patches, bin_centers):
    color = cmap(norm(x_value))  # Map x-value to a color
    patch.set_facecolor(color)

ax2.set_xlabel("Coefficients of variation (%)", fontsize = fs+11)
ax2.set_ylabel("Density", fontsize = fs+11)
ax2.set_xlim(0, 30)

ax2.set_yticks([])
ax2.tick_params(axis='x', which='major', labelsize = fs+11)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig("GMD_illustrations/CV_"+strh+"_"+str_dist+"_final_"+str(valid_freq)+"_alt_b.png", dpi=my_dpi)