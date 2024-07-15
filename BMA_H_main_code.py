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
 
rgb_misc = np.loadtxt("GMD_data_input/misc_div.txt") # downloadable at https://github.com/IPCC-WG1/colormaps
rgb_slev = np.loadtxt("GMD_data_input/slev_div.txt") # downloadable at https://github.com/IPCC-WG1/colormaps

#create the colormap
IPCC_misc = colors.LinearSegmentedColormap.from_list('colormap', rgb_misc)
IPCC_slev = colors.LinearSegmentedColormap.from_list('colormap', rgb_slev)
 
my_dpi = 500

fs = 9
 
## Data preparation 

NFI_ind = 7 # the type of the NFI plot height employed

if NFI_ind == 1:
	str = "H_max" # maximum height without imputations
elif NFI_ind == 2:
	str = "H_mean" # mean height without imputations
elif NFI_ind == 3:
	str = "H_dom" # dominant height without imputations
elif NFI_ind == 4:
	str = "Hi_max" # maximum height with imputations
elif NFI_ind == 5:
	str = "Hi_mean" # mean height with imputations
elif NFI_ind == 6:
	str = "Hi_dom" # dominant height with imputations
elif NFI_ind == 7:
	str = "Hi_lor" # Loray's height with imputations
	
	
# Reading the csv file contaning the NFI plot heights along with other variables, as well as
# the extractions from 5 diffent models (maps) at these plots.

# This file is not available for download, due to the statistical secret applied on the exact position of plots,
# but the confidential access could be provided for the editor and reviewers in order to enable peer review.

data = pd.read_csv("GMD_data_input/Input_data_table.csv")

I=np.zeros((data.shape[0], 6))

I[:,0]=data["M_1"].to_numpy() # Lang 2020 (m) - available for download at the link provided in the manuscript
I[:,1]=data["M_2"].to_numpy() # Liu 2019 (m) - available for download at the link provided in the manuscript
I[:,2]=data["M_3"].to_numpy()/100 # Morin 2020 (cm) - available for download at the link provided in the manuscript
I[:,3]=data["M_4"].to_numpy() # Potapov 2019 (m) - available for download at the link provided in the manuscript
I[:,4]=data["M_5"].to_numpy()/100 # Schwartz 2020 (cm) - available for download at the link provided in the manuscript

I[:,5]=data[str].to_numpy() # selected NFI height reference

H_all = pd.DataFrame()

H_all["Lang"] = I[:,0]
H_all["Liu"] = I[:,1]
H_all["Morin"] = I[:,2]
H_all["Potapov"] = I[:,3]
H_all["Schwartz"] = I[:,4]
H_all["NFI"] = I[:,5]

### Bayesian model averaging

epsilon = 1e-6 # EM algorithm convergance parameters
maxiter = 1e5 # EM algorithm convergance parameters

pdf_ind = 1 # 1- normal, 2 - lognormal

if pdf_ind == 1:
	str_dist = "norm"
elif pdf_ind == 2:
	str_dist = "log_norm"

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

def py_sweep(a,b):

    return a*b

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

H_all["BMA"] = np.nansum(I[:,0:5] * z_new_1, axis = 1)

# Between-model spread

var_inter = np.nansum(z_new * (I[:, range(K)] - np.tile(np.nansum(z_new * I[:, range(K)], axis = 1)[np.newaxis].T, [1, K]))**2, axis = 1)

var_inter_1 = np.nansum(z_1 * (I[:, range(K)] - np.tile(np.nansum(z_1 * I[:, range(K)], axis = 1)[np.newaxis].T, [1, K]))**2, axis = 1)

# Within-model variance

var_intra = np.nansum(z_new * np.tile(sigma_new**2, [N, 1]), axis = 1)

var_intra_1 = np.nansum(z_1 * np.tile(sigma_new**2, [N, 1]), axis = 1)

np.save("GMD_data_output/Weights_"+str+"_"+str_dist+".npy", w_new)
np.save("GMD_data_output/Local_Weights_"+str+"_"+str_dist+".npy", z_new)
np.save("GMD_data_output/Var_inter_"+str+"_"+str_dist+".npy", var_inter)
np.save("GMD_data_output/Var_intra_"+str+"_"+str_dist+".npy", var_intra)

## Sample size by SER (French sylvo-ecological regions)

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
ser=spec.to_numpy()

SER_mat=np.zeros((len(ser),86))

for i in range(len(ser)):
	for j in range(86):
		if ser[i] == j:
			SER_mat[i,j]+=1

SER_mat_nb = np.sum(SER_mat, axis=0)

ser_mask = gpd.read_file("GMD_data_input/ser_l93.shp") # available for download at the link https://inventaire-forestier.ign.fr/spip.php?article729 (in French) 

ser_mask = ser_mask.drop([149, 150])

spec1 = ser_mask['codeser'].map(SERcls)
ser1 = spec1.to_numpy()

ser1 = ser1.astype(int)

ser_mask["area"] = ser_mask['geometry'].area
ser_mask["nb_sample"] = SER_mat_nb[ser1]
ser_mask["norm_nb_sample"] = SER_mat_nb[ser1]/ser_mask['geometry'].area

## Illustrations - sample size by SER (not illustrated in the manuscript)

fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot2grid((1, 3), (0, 0))

ser_mask.plot(ax = ax1, column = 'area', cmap = "summer")

ax1.set_xticks([])
ax1.set_yticks([])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.set_title("SER areas", fontsize=fs-2)

ax2 = plt.subplot2grid((1, 3), (0, 1))

ser_mask.plot(ax = ax2, column = 'nb_sample', cmap = "summer")

ax2.set_xticks([])
ax2.set_yticks([])

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.set_title("SER sample size", fontsize=fs-2)

ax3 = plt.subplot2grid((1, 3), (0, 2))

ser_mask.plot(ax = ax3, column = 'norm_nb_sample', cmap = "summer", vmin = ser_mask["norm_nb_sample"].min(), vmax = ser_mask["norm_nb_sample"].max())

ax3.set_xticks([])
ax3.set_yticks([])

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

ax3.set_title("SER sample size / area", fontsize=fs-2)

plt.savefig("GMD_illustrations/SER_nb_samples.png", dpi=my_dpi)

## M1 weights by SER

M1_mat=np.zeros((np.shape(z_new)[0],86))

for i in range(np.shape(z_new)[0]):
	for j in range(86):
		if ser[i] == j:
			M1_mat[i,j] += z_new[i,0]

SER_mat_nb[SER_mat_nb == 0] = np.nan

M1_mat_mean = np.nansum(M1_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(M1_mat) == 1, axis=0))

## M2 weights by SER

M2_mat=np.zeros((np.shape(z_new)[0],86))

for i in range(np.shape(z_new)[0]):
	for j in range(86):
		if ser[i] == j:
			M2_mat[i,j] += z_new[i,1]

M2_mat_mean = np.nansum(M2_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(M2_mat) == 1, axis=0))

## M3 weights by SER

M3_mat=np.zeros((np.shape(z_new)[0],86))

for i in range(np.shape(z_new)[0]):
	for j in range(86):
		if ser[i] == j:
			M3_mat[i,j] += z_new[i,2]

M3_mat_mean = np.nansum(M3_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(M3_mat) == 1, axis=0))

## M4 weights by SER

M4_mat=np.zeros((np.shape(z_new)[0],86))

for i in range(np.shape(z_new)[0]):
	for j in range(86):
		if ser[i] == j:
			M4_mat[i,j] += z_new[i,3]

M4_mat_mean = np.nansum(M4_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(M4_mat) == 1, axis=0))

## M5 weights by SER

M5_mat=np.zeros((np.shape(z_new)[0],86))

for i in range(np.shape(z_new)[0]):
	for j in range(86):
		if ser[i] == j:
			M5_mat[i,j] += z_new[i,4]

M5_mat_mean = np.nansum(M5_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(M5_mat) == 1, axis=0))

## Var_inter by SER

Var_inter_mat=np.zeros((len(var_inter),86))

for i in range(len(var_inter)):
	for j in range(86):
		if ser[i] == j:
			Var_inter_mat[i,j] += var_inter[i]

Var_inter_mat_mean = np.nansum(Var_inter_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(Var_inter_mat) == 1, axis=0))

Var_inter_1_mat=np.zeros((len(var_inter_1),86))

for i in range(len(var_inter_1)):
	for j in range(86):
		if ser[i] == j:
			Var_inter_1_mat[i,j] += var_inter_1[i]

Var_inter_1_mat_mean = np.nansum(Var_inter_1_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(Var_inter_1_mat) == 1, axis=0))

## Var_intra by SER

Var_intra_mat=np.zeros((len(var_intra),86))

for i in range(len(var_intra)):
	for j in range(86):
		if ser[i] == j:
			Var_intra_mat[i,j] += var_intra[i]

Var_intra_mat_mean = np.nansum(Var_intra_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(Var_intra_mat) == 1, axis=0))

Var_intra_1_mat=np.zeros((len(var_intra_1),86))

for i in range(len(var_intra_1)):
	for j in range(86):
		if ser[i] == j:
			Var_intra_1_mat[i,j] += var_intra_1[i]

Var_intra_1_mat_mean = np.nansum(Var_intra_1_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(Var_intra_1_mat) == 1, axis=0))

M_tout = np.array([M1_mat_mean[ser1], M2_mat_mean[ser1], M3_mat_mean[ser1], M4_mat_mean[ser1], M5_mat_mean[ser1]]).T

M_sum = np.sum(M_tout, axis = 1)

M_dom = np.argmax(M_tout, axis = 1) + 1

M_dom = M_dom.astype(float)

M_dom[np.isnan(M_sum) == 1] = np.nan

ser_mask["M1_weights"] = M_tout[:,0] / M_sum 
ser_mask["M2_weights"] = M_tout[:,1] / M_sum 
ser_mask["M3_weights"] = M_tout[:,2] / M_sum 
ser_mask["M4_weights"] = M_tout[:,3] / M_sum 
ser_mask["M5_weights"] = M_tout[:,4] / M_sum 

ser_mask["M_dominant"] = M_dom

ser_mask["Within_variance"] = Var_intra_mat_mean[ser1]
ser_mask["Between_variance"] = Var_inter_mat_mean[ser1]
ser_mask["Within-between"] = Var_intra_mat_mean[ser1] - Var_inter_mat_mean[ser1]

ser_mask["Within_variance_1"] = Var_intra_1_mat_mean[ser1]
ser_mask["Between_variance_1"] = Var_inter_1_mat_mean[ser1]
ser_mask["Within-between_1"] = Var_intra_1_mat_mean[ser1] - Var_inter_1_mat_mean[ser1]

## Illustration - local (regional) weights (Fig. 3 in the manuscript)

fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot2grid((1, 13), (0, 0), colspan = 2)

cmap1 = plt.cm.Blues

ser_mask.plot(ax = ax1, column = 'M1_weights', cmap = cmap1)

# norm = colors.Normalize(vmin = 0, vmax = ser_mask['M1_weights'].max())
# cbar = plt.cm.ScalarMappable(norm = norm, cmap = cmap1)
# ax_cbar = fig.colorbar(cbar, ax = ax1, shrink = 0.85, ticks=[0., ser_mask['M1_weights'].max()], location = "top")
# ax_cbar.ax.set_xticklabels(['$0$', '$w_{\mathrm{SER}}^{\max}$'])
# ax_cbar.ax.tick_params(labelsize = fs-2, top = True, labeltop = True)

ax1.set_xticks([])
ax1.set_yticks([])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.set_title("$M_1$: $w_{\mathrm{SER}}$ ($w_{\mathrm{SER}}^{\max}$ = %.2f)" % ser_mask['M1_weights'].max(), fontsize = fs-2)

ax2 = plt.subplot2grid((1, 13), (0, 2), colspan = 2)

cmap2 = plt.cm.Purples

ser_mask.plot(ax = ax2, column = 'M2_weights', cmap = cmap2)

# norm = colors.Normalize(vmin = 0, vmax = ser_mask['M2_weights'].max())
# cbar = plt.cm.ScalarMappable(norm = norm, cmap = cmap2)
# ax_cbar = fig.colorbar(cbar, ax = ax2, shrink = 0.85, ticks=[0., ser_mask['M2_weights'].max()], location = "top")
# ax_cbar.ax.set_xticklabels(['$0$', '$w_{\mathrm{SER}}^{\max}$'])
# ax_cbar.ax.tick_params(labelsize = fs-2, top = True, labeltop = True)

ax2.set_xticks([])
ax2.set_yticks([])

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.set_title("$M_2$: $w_{\mathrm{SER}}$ ($w_{\mathrm{SER}}^{\max}$ = %.2f)" % ser_mask['M2_weights'].max(), fontsize = fs-2)

ax3 = plt.subplot2grid((1, 13), (0, 4), colspan = 2)

cmap3 = plt.cm.Reds

ser_mask.plot(ax = ax3, column = 'M3_weights', cmap = cmap3)

# norm = colors.Normalize(vmin = 0, vmax = ser_mask['M3_weights'].max())
# cbar = plt.cm.ScalarMappable(norm = norm, cmap = cmap3)
# ax_cbar = fig.colorbar(cbar, ax = ax3, shrink = 0.85, ticks=[0., ser_mask['M3_weights'].max()], location = "top")
# ax_cbar.ax.set_xticklabels(['$0$', '$w_{\mathrm{SER}}^{\max}$'])
# ax_cbar.ax.tick_params(labelsize = fs-2, top = True, labeltop = True)


ax3.set_xticks([])
ax3.set_yticks([])

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

ax3.set_title("$M_3$: $w_{\mathrm{SER}}$ ($w_{\mathrm{SER}}^{\max}$ = %.2f)" % ser_mask['M3_weights'].max(), fontsize = fs-2)

ax4 = plt.subplot2grid((1, 13), (0, 6), colspan = 2)

cmap4 = plt.cm.Oranges

ser_mask.plot(ax = ax4, column = 'M4_weights', cmap = cmap4)

# norm = colors.Normalize(vmin = 0, vmax = ser_mask['M4_weights'].max())
# cbar = plt.cm.ScalarMappable(norm = norm, cmap = cmap4)
# ax_cbar = fig.colorbar(cbar, ax = ax4, shrink = 0.85, ticks=[0., ser_mask['M4_weights'].max()], location = "top")
# ax_cbar.ax.set_xticklabels(['$0$', '$w_{\mathrm{SER}}^{\max}$'])
# ax_cbar.ax.tick_params(labelsize = fs-2, top = True, labeltop = True)

ax4.set_xticks([])
ax4.set_yticks([])

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)

ax4.set_title("$M_4$: $w_{\mathrm{SER}}$ ($w_{\mathrm{SER}}^{\max}$ = %.2f)" % ser_mask['M4_weights'].max(), fontsize = fs-2)

ax5 = plt.subplot2grid((1, 13), (0, 8), colspan = 2)

cmap5 = plt.cm.Greens

ser_mask.plot(ax = ax5, column = 'M5_weights', cmap = cmap5, legend = False)

# norm = colors.Normalize(vmin = 0, vmax = ser_mask['M5_weights'].max())
# cbar = plt.cm.ScalarMappable(norm = norm, cmap = cmap5)
# ax_cbar = fig.colorbar(cbar, ax = ax5, shrink = 0.85, ticks=[0., ser_mask['M5_weights'].max()], location = "top")
# ax_cbar.ax.set_xticklabels(['$0$', '$w_{\mathrm{SER}}^{\max}$'])
# ax_cbar.ax.tick_params(labelsize = fs-2, top = True, labeltop = True)

ax5.set_xticks([])
ax5.set_yticks([])

ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.spines['left'].set_visible(False)

ax5.set_title("$M_5$: $w_{\mathrm{SER}}$ ($w_{\mathrm{SER}}^{\max}$ = %.2f)" % ser_mask['M5_weights'].max(), fontsize = fs-2)

ax6 = plt.subplot2grid((1, 13), (0, 10), colspan = 3)

cmap_colors = np.array([[0.23192618223760095, 0.5456516724336793, 0.7626143790849673, 1.0],
						[0.48402921953094963, 0.4543329488658208, 0.7106651287966167, 1.0],
					    [0.907912341407151, 0.20284505959246443, 0.16032295271049596, 1.0],
					    [0.9255363321799308, 0.3848673587081891, 0.05983852364475202, 1.0],
						[0.23044982698961936, 0.6445059592464436, 0.34514417531718566, 1.0]])

custom_cmap= ListedColormap(cmap_colors)

ser_mask.plot(ax = ax6, column = 'M_dominant', cmap = custom_cmap, legend = False, vmin = 0.5, vmax = 5.5)

# norm = colors.Normalize(vmin = 0.5, vmax = 5.5)
# cbar = plt.cm.ScalarMappable(norm = norm, cmap = custom_cmap)
# ax_cbar = fig.colorbar(cbar, ax = ax6, shrink = 0.85, ticks=[1., 2., 3., 4., 5.], location = "top")
# ax_cbar.ax.set_xticklabels(['$M_1$', '$M_2$', '$M_3$', '$M_4$', '$M_5$'])
# ax_cbar.ax.tick_params(labelsize = fs-2, top = True, labeltop = True)

ax6.set_xticks([])
ax6.set_yticks([])

ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.spines['left'].set_visible(False)

# if np.any(NFI_ind == np.array([1, 2, 3])):
ax6.set_title("Dominant model by SER", fontsize=fs-2)

plt.tight_layout(pad = 0)

plt.savefig("GMD_illustrations/SER_weights_dom_"+str+"_"+str_dist+".png", dpi=my_dpi)

## Illustration - BMA vs. SMA (Fig. 6 in the manuscript)

fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot2grid((2, 3), (0, 0))

cmap1 = IPCC_misc

V_min = ser_mask['Within_variance'].min()
V_max = ser_mask['Within_variance'].max()
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax1, column = 'Within_variance', cmap = cmap1, legend = True, legend_kwds={"shrink": 0.5, "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax1.set_xticks([])
ax1.set_yticks([])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.set_title("Within variance (BMA)", fontsize = fs-1)

ax2 = plt.subplot2grid((2, 3), (0, 1))

cmap2 = IPCC_misc

V_min = ser_mask['Between_variance'].min()
V_max = ser_mask['Between_variance'].max()
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax2, column = 'Between_variance', cmap = cmap2, legend = True, legend_kwds={"shrink": 0.5, "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax2.set_xticks([])
ax2.set_yticks([])

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.set_title("Between variance (BMA)", fontsize = fs-1)

ax3 = plt.subplot2grid((2, 3), (0, 2))

cmap3 = IPCC_misc

V_min = np.min([ser_mask['Within-between'].min(),ser_mask['Within-between_1'].min()])
V_max = np.max([ser_mask['Within-between'].max(),ser_mask['Within-between_1'].max()])
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax3, column = 'Within-between', cmap = cmap3, legend = True, legend_kwds={"shrink": 0.5, "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax3.set_xticks([])
ax3.set_yticks([])

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

ax3.set_title("Within $-$ between (BMA)", fontsize = fs-1)

ax4 = plt.subplot2grid((2, 3), (1, 0))

cmap4 = IPCC_misc

V_min = ser_mask['Within_variance_1'].min()
V_max = ser_mask['Within_variance_1'].max()
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax4, column = 'Within_variance_1', cmap = cmap4, legend = True, legend_kwds={"shrink": 0.5, "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax4.set_xticks([])
ax4.set_yticks([])

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)

ax4.set_title("Within variance (SMA)", fontsize = fs-1)

ax5 = plt.subplot2grid((2, 3), (1, 1))

cmap5 = IPCC_misc

V_min = ser_mask['Between_variance_1'].min()
V_max = ser_mask['Between_variance_1'].max()
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax5, column = 'Between_variance_1', cmap = cmap5, legend = True, legend_kwds={"shrink": 0.5, "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax5.set_xticks([])
ax5.set_yticks([])

ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.spines['left'].set_visible(False)

ax5.set_title("Between variance (SMA)", fontsize = fs-1)

ax6 = plt.subplot2grid((2, 3), (1, 2))

cmap6 = IPCC_misc

V_min = np.min([ser_mask['Within-between'].min(),ser_mask['Within-between_1'].min()])
V_max = np.max([ser_mask['Within-between'].max(),ser_mask['Within-between_1'].max()])
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax6, column = 'Within-between_1', cmap = cmap6, legend = True, legend_kwds={"shrink": 0.5, "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax6.set_xticks([])
ax6.set_yticks([])

ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.spines['left'].set_visible(False)

ax6.set_title("Within $-$ between (SMA)", fontsize = fs-1)

plt.savefig("GMD_illustrations/Variances_"+str+"_"+str_dist+".png", dpi=my_dpi)

## Illustration - individual models vs. BMA (Fig. 5 in the manuscript)

fig = plt.figure(figsize=(15,6))

ax1 = plt.subplot2grid((2, 6), (0, 0))

im1 = sns.kdeplot(data = H_all, x = "Lang", y = "NFI", color = (0.23192618223760095, 0.5456516724336793, 0.7626143790849673), legend = False)

ax1.plot([0, np.nanmax(H_all["NFI"])], [0, np.nanmax(H_all["NFI"])], color = "k", linestyle = 'dashed', linewidth = 1)

H_all_1 = H_all.copy()
H_all_1 = H_all_1.dropna(subset=['NFI', 'Lang'])

ax1.set_xlabel("$M_1$",fontsize = fs)
ax1.set_ylabel("$NFI$",fontsize = fs)
ax1.set_title("$M_1$: $R^2=$%.2f,\n $MBE=$%.2f, $RMSE=$%.2f" % 
			  (r2_score(H_all_1["NFI"], H_all_1["Lang"]), np.mean(H_all_1["Lang"] - H_all_1["NFI"]),
	  		   np.sqrt(mean_squared_error(H_all_1["NFI"], H_all_1["Lang"]))),fontsize = fs)

plt.xticks(fontsize = fs-1)
plt.yticks(fontsize = fs-1)

ax1.set_xlim(0,np.nanmax(H_all["NFI"]))
ax1.set_ylim(0,np.nanmax(H_all["NFI"]))

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax2 = plt.subplot2grid((2, 6), (0, 1))

im2 = sns.kdeplot(data = H_all, x = "Liu", y = "NFI", color = (0.48402921953094963, 0.4543329488658208, 0.7106651287966167), legend = False)

ax2.plot([0, np.nanmax(H_all["NFI"])], [0, np.nanmax(H_all["NFI"])], color = "k", linestyle = 'dashed', linewidth = 1)

H_all_1 = H_all.copy()
H_all_1 = H_all_1.dropna(subset=['NFI', 'Liu'])

ax2.set_xlabel("$M_2$",fontsize = fs)
ax2.set_ylabel("$NFI$",fontsize = fs)
ax2.set_title("$M_2$: $R^2=$%.2f,\n $MBE=$%.2f, $RMSE=$%.2f" % 
			  (r2_score(H_all_1["NFI"], H_all_1["Liu"]), np.mean(H_all_1["Liu"] - H_all_1["NFI"]),
	  		   np.sqrt(mean_squared_error(H_all_1["NFI"], H_all_1["Liu"]))),fontsize = fs)

plt.xticks(fontsize = fs-1)
plt.yticks(fontsize = fs-1)

ax2.set_xlim(0,np.nanmax(H_all["NFI"]))
ax2.set_ylim(0,np.nanmax(H_all["NFI"]))

ax2.set_yticks([])
ax2.set_ylabel("")

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax3 = plt.subplot2grid((2, 6), (0, 2))

im3 = sns.kdeplot(data = H_all, x = "Morin", y = "NFI", color = (0.907912341407151, 0.20284505959246443, 0.16032295271049596), legend = False)

ax3.plot([0, np.nanmax(H_all["NFI"])], [0, np.nanmax(H_all["NFI"])], color = "k", linestyle = 'dashed', linewidth = 1)

H_all_1 = H_all.copy()
H_all_1 = H_all_1.dropna(subset=['NFI', 'Morin'])

ax3.set_xlabel("$M_3$",fontsize = fs)
ax3.set_ylabel("$NFI$",fontsize = fs)
ax3.set_title("$M_3$: $R^2=$%.2f,\n $MBE=$%.2f, $RMSE=$%.2f" % 
			  (r2_score(H_all_1["NFI"], H_all_1["Morin"]), np.mean(H_all_1["Morin"] - H_all_1["NFI"]),
	  		   np.sqrt(mean_squared_error(H_all_1["NFI"], H_all_1["Morin"]))),fontsize = fs)

plt.xticks(fontsize = fs-1)
plt.yticks(fontsize = fs-1)

ax3.set_xlim(0,np.nanmax(H_all["NFI"]))
ax3.set_ylim(0,np.nanmax(H_all["NFI"]))

ax3.set_yticks([])
ax3.set_ylabel("")

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

ax4 = plt.subplot2grid((2, 6), (0, 3))

im4 = sns.kdeplot(data = H_all, x = "Potapov", y = "NFI", color = (0.9255363321799308, 0.3848673587081891, 0.05983852364475202), legend = False)

ax4.plot([0, np.nanmax(H_all["NFI"])], [0, np.nanmax(H_all["NFI"])], color = "k", linestyle = 'dashed', linewidth = 1)

H_all_1 = H_all.copy()
H_all_1 = H_all_1.dropna(subset=['NFI', 'Potapov'])

ax4.set_xlabel("$M_4$",fontsize = fs)
ax4.set_ylabel("$NFI$",fontsize = fs)
ax4.set_title("$M_4$: $R^2=$%.2f,\n $MBE=$%.2f, $RMSE=$%.2f" % 
			  (r2_score(H_all_1["NFI"], H_all_1["Potapov"]), np.mean(H_all_1["Potapov"] - H_all_1["NFI"]),
	  		   np.sqrt(mean_squared_error(H_all_1["NFI"], H_all_1["Potapov"]))),fontsize = fs)

plt.xticks(fontsize = fs-1)
plt.yticks(fontsize = fs-1)

ax4.set_xlim(0,np.nanmax(H_all["NFI"]))
ax4.set_ylim(0,np.nanmax(H_all["NFI"]))

ax4.set_yticks([])
ax4.set_ylabel("")

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)

ax5 = plt.subplot2grid((2, 6), (0, 4))

im5 = sns.kdeplot(data = H_all, x = "Schwartz", y = "NFI", color = (0.23044982698961936, 0.6445059592464436, 0.34514417531718566), legend = False)

ax5.plot([0, np.nanmax(H_all["NFI"])], [0, np.nanmax(H_all["NFI"])], color = "k", linestyle = 'dashed', linewidth = 1)

H_all_1 = H_all.copy()
H_all_1 = H_all_1.dropna(subset=['NFI', 'Schwartz'])

ax5.set_xlabel("$M_5$",fontsize = fs)
ax5.set_ylabel("$NFI$",fontsize = fs)
ax5.set_title("$M_5$: $R^2=$%.2f,\n $MBE=$%.2f, $RMSE=$%.2f" % 
			  (r2_score(H_all_1["NFI"], H_all_1["Schwartz"]), np.mean(H_all_1["Schwartz"] - H_all_1["NFI"]),
	  		   np.sqrt(mean_squared_error(H_all_1["NFI"], H_all_1["Schwartz"]))),fontsize = fs)

plt.xticks(fontsize = fs-1)
plt.yticks(fontsize = fs-1)

ax5.set_xlim(0,np.nanmax(H_all["NFI"]))
ax5.set_ylim(0,np.nanmax(H_all["NFI"]))

ax5.set_yticks([])
ax5.set_ylabel("")

ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.spines['left'].set_visible(False)

ax6 = plt.subplot2grid((2, 6), (0, 5))

im6 = sns.kdeplot(data = H_all, x = "BMA", y = "NFI", color = "k", legend = False)

ax6.plot([0, np.nanmax(H_all["NFI"])], [0, np.nanmax(H_all["NFI"])], color = "k", linestyle = 'dashed', linewidth = 1)

H_all_1 = H_all.copy()
H_all_1 = H_all_1.dropna(subset=['NFI', 'BMA'])

ax6.set_xlabel("$BMA$",fontsize = fs)
ax6.set_ylabel("$NFI$",fontsize = fs)
ax6.set_title("$BMA$: $R^2=$%.2f,\n $MBE=$%.2f, $RMSE=$%.2f" % 
			  (r2_score(H_all_1["NFI"], H_all_1["BMA"]), np.mean(H_all_1["BMA"] - H_all_1["NFI"]),
	  		   np.sqrt(mean_squared_error(H_all_1["NFI"], H_all_1["BMA"]))),fontsize = fs)

plt.xticks(fontsize = fs-1)
plt.yticks(fontsize = fs-1)

ax6.set_xlim(0,np.nanmax(H_all["NFI"]))
ax6.set_ylim(0,np.nanmax(H_all["NFI"]))

ax6.set_yticks([])
ax6.set_ylabel("")

ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.spines['left'].set_visible(False)

plt.savefig("GMD_illustrations/Scatterplot_"+str+"_"+str_dist+".png", dpi=my_dpi)

MNT_elev = data["DTM_elevation"].to_numpy() # DTM elevation
MNT_pente = data["DTM_slope"].to_numpy() # DTM slope
MNT_exp = data["DTM_aspect"].to_numpy() # DTM aspect

FR = data["Tree_type"].to_numpy() # broad-leaved or coniferous
Ess = data["Tree_species"].to_numpy() # tree essence
SVER = data["Vertical_structure"].to_numpy() # broad-leaved or coniferous
PROP = data["Property_type"].to_numpy() # tree essence

Analyses = pd.DataFrame()

Analyses["Within_variance"] = var_intra
Analyses["Between_variance"] = var_inter

Analyses["Tree_dominant_type"] = FR
Analyses["Tree_dominant_species"] = Ess
Analyses["SVER"] = SVER
Analyses["PROP"] = PROP

## Influence of categorical variables on the spread (Table 1 in the manuscript)

# Ordinary Least Squares (OLS) model
model5 = ols('Between_variance ~ C(Tree_dominant_type) + C(Tree_dominant_species) + C(SVER)+ C(PROP)', data=Analyses).fit()

anova_table5 = sm.stats.anova_lm(model5, typ=2)
print(anova_table5)

## Illustration - influence of categorical variables on the spread (Fig. A1 in the manuscript)

fig = plt.figure(figsize=(20,6))

ax1 = plt.subplot2grid((2, 6), (0, 0))

im1 = sns.boxplot(x = "SVER", y = "Between_variance", showfliers = False,  palette = "plasma", data = Analyses)

plt.xticks(fontsize = fs-2)
plt.yticks(fontsize = fs-2)

Labels = np.array([])

for label in im1.get_xticklabels():
	Labels = np.append (Labels, label.get_text())

xlabels = ['V' + '{:,.0f}'.format(x) for x in Labels.astype(float)]
im1.set_xticklabels(xlabels)

ax1.set_xlabel('Vertical structure', fontsize = fs-1) 
ax1.set_ylabel('Between variance', fontsize = fs-1) 

ax2 = plt.subplot2grid((2, 6), (0, 1))

im2 = sns.boxplot(x = "PROP", y = "Between_variance", showfliers = False,  palette = "viridis", data = Analyses)

plt.xticks(fontsize = fs-2)
plt.yticks(fontsize = fs-2)

Labels = np.array([])

for label in im2.get_xticklabels():
	Labels = np.append (Labels, label.get_text())

xlabels = ['O' + '{:,.0f}'.format(x) for x in Labels.astype(float)]
im2.set_xticklabels(xlabels)

ax2.set_xlabel('Type of ownership', fontsize = fs-1) 
ax2.set_ylabel('', fontsize = fs-1) 

ax3 = plt.subplot2grid((2, 6), (0, 2), colspan = 4)

group_means = Analyses.groupby(['Tree_dominant_species'])['Between_variance'].mean().sort_values(ascending = False)

im3 = sns.boxplot(x = "Tree_dominant_species", y = "Between_variance", palette = "cividis", showfliers = False, data = Analyses, order=group_means.index)

plt.xticks(fontsize = fs-2, rotation=40)
plt.yticks(fontsize = fs-2)

Labels = np.array([])

for label in im3.get_xticklabels():
	Labels = np.append (Labels, label.get_text())

xlabels = ['S' + '{:,.0f}'.format(x) for x in Labels.astype(float)]
im3.set_xticklabels(xlabels)

ax3.set_xlabel('Dominant tree species', fontsize = fs-1) 
ax3.set_ylabel('', fontsize = fs-1) 

plt.savefig("Illustrations/Boxplots_"+str+"_"+str_dist+".png", dpi=my_dpi)

corr_plot = 0 # printing correlations (1) or not (any other value)

if corr_plot == 1:

	print("Corr_MNT_elev_inter")

	Corr_MNT_elev_inter = np.corrcoef(var_inter, MNT_elev)

	print(Corr_MNT_elev_inter[0,1])

	print("Corr_MNT_pente_inter")

	a=ma.masked_invalid(var_inter)
	b=ma.masked_invalid(MNT_pente)

	msk = (~a.mask & ~b.mask)

	Corr_MNT_pente_inter = np.ma.corrcoef(a[msk], b[msk])

	print(Corr_MNT_pente_inter[0,1])

	print("Corr_MNT_pente_inter")

	a=ma.masked_invalid(var_inter)
	b=ma.masked_invalid(MNT_exp)

	msk = (~a.mask & ~b.mask)

	Corr_MNT_exp_inter = np.ma.corrcoef(a[msk], b[msk])

	print(Corr_MNT_exp_inter[0,1])

	print("Corr_MNT_elev_intra")

	Corr_MNT_elev_intra = np.corrcoef(var_intra, MNT_elev)

	print(Corr_MNT_elev_intra[0,1])

	print("Corr_MNT_pente_intra")

	a=ma.masked_invalid(var_intra)
	b=ma.masked_invalid(MNT_pente)

	msk = (~a.mask & ~b.mask)

	Corr_MNT_pente_intra = np.ma.corrcoef(a[msk], b[msk])

	print(Corr_MNT_pente_intra[0,1])

	print("Corr_MNT_pente_intra")

	a=ma.masked_invalid(var_intra)
	b=ma.masked_invalid(MNT_exp)

	msk = (~a.mask & ~b.mask)

	Corr_MNT_exp_intra = np.ma.corrcoef(a[msk], b[msk])

	print(Corr_MNT_exp_intra[0,1])

# Mean elev by SER

MNT_elev_mat=np.zeros((len(MNT_elev),86))

for i in range(len(MNT_elev)):
	for j in range(86):
		if ser[i] == j:
			MNT_elev_mat[i,j] += MNT_elev[i]

MNT_elev_mat_mean = np.nansum(MNT_elev_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(MNT_elev_mat) == 1, axis=0))

# Mean slope by SER

MNT_pente_mat=np.zeros((len(MNT_pente),86))

for i in range(len(MNT_pente)):
	for j in range(86):
		if ser[i] == j:
			MNT_pente_mat[i,j] += MNT_pente[i]

MNT_pente_mat_mean = np.nansum(MNT_pente_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(MNT_pente_mat) == 1, axis=0))

# Mean aspect by SER

MNT_exp_mat=np.zeros((len(MNT_exp),86))

for i in range(len(MNT_exp)):
	for j in range(86):
		if ser[i] == j:
			MNT_exp_mat[i,j] += MNT_exp[i]

MNT_exp_mat_mean = np.nansum(MNT_exp_mat, axis=0) / (SER_mat_nb - np.sum(np.isnan(MNT_exp_mat) == 1, axis=0))

## Illustration

ser_mask["MNT_elev"] = MNT_elev_mat_mean[ser1]
ser_mask["MNT_pente"] = MNT_pente_mat_mean[ser1]
ser_mask["MNT_exp"] = MNT_exp_mat_mean[ser1]

## Illustration - influence of topography on the spread (Fig. 4e in the manuscript)

fig = plt.figure(figsize=(10,6))

ax0 = plt.subplot2grid((1, 4), (0, 0))

cmap1 = IPCC_slev

V_min = ser_mask['MNT_elev'].min()
V_max = ser_mask['MNT_elev'].max()
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax0, column = 'MNT_elev', cmap = cmap1, legend = True, legend_kwds={"shrink": 0.2, "format": "%.1f", "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax0.set_xticks([])
ax0.set_yticks([])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)

ax0.set_title("DTM mean elevation by SER", fontsize=fs-2)

ax1 = plt.subplot2grid((1, 4), (0, 2))

cmap1 = IPCC_slev

V_min = ser_mask['MNT_elev'].min()
V_max = ser_mask['MNT_elev'].max()
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax1, column = 'MNT_elev', cmap = cmap1, legend = True, legend_kwds={"shrink": 0.2, "format": "%.1f", "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax1.set_xticks([])
ax1.set_yticks([])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.set_title("DTM mean elevation by SER", fontsize=fs-2)

ax2 = plt.subplot2grid((1, 4), (0, 3))

cmap1 = IPCC_slev

V_min = ser_mask['MNT_pente'].min()
V_max = ser_mask['MNT_pente'].max()
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax2, column = 'MNT_pente', cmap = cmap1, legend = True, legend_kwds={"shrink": 0.2, "format": "%.1f", "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax2.set_xticks([])
ax2.set_yticks([])

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.set_title("DTM mean slope by SER", fontsize=fs-2)

plt.tight_layout(pad = 2)

plt.savefig("Illustrations/MNT_by_SER.png", dpi=my_dpi)

print("Corr_MNT_elev_SER_inter")

a=ma.masked_invalid(Var_inter_mat_mean)
b=ma.masked_invalid(MNT_elev_mat_mean)

msk = (~a.mask & ~b.mask)

Corr_MNT_elev_SER_inter = np.ma.corrcoef(a[msk], b[msk])

print(Corr_MNT_elev_SER_inter[0,1])

print("Corr_MNT_pente_SER_inter")

a=ma.masked_invalid(Var_inter_mat_mean)
b=ma.masked_invalid(MNT_pente_mat_mean)

msk = (~a.mask & ~b.mask)

Corr_MNT_pente_SER_inter = np.ma.corrcoef(a[msk], b[msk])

print(Corr_MNT_pente_SER_inter[0,1])

## Illustration - influence of topography on the spread (Fig. 4a - 4d in the manuscript)

fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot2grid((1, 4), (0, 0))

cmap1 = IPCC_misc

V_min = ser_mask['Within_variance'].min()
V_max = ser_mask['Within_variance'].max()
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax1, column = 'Within_variance', cmap = cmap1, legend = True, legend_kwds={"shrink": 0.2, "format": "%.1f", "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax1.set_xticks([])
ax1.set_yticks([])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.set_title("Within variance by SER", fontsize = fs-2)

ax2 = plt.subplot2grid((1, 4), (0, 1))

cmap2 = IPCC_misc

V_min = ser_mask['Between_variance'].min()
V_max = ser_mask['Between_variance'].max()
V_mean = V_min + (V_max-V_min)/2

ser_mask.plot(ax = ax2, column = 'Between_variance', cmap = cmap2, legend = True, legend_kwds={"shrink": 0.2, "format": "%.1f", "ticks": [V_min, V_mean, V_max]}, 
				vmin = V_min,
				vmax = V_max)

ax2.set_xticks([])
ax2.set_yticks([])

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.set_title("Between variance by SER", fontsize = fs-2)

ax3 = plt.subplot2grid((1, 4), (0, 2))

im3 = ax3.plot([np.nanmin(MNT_elev_mat_mean), np.nanmax(MNT_elev_mat_mean)], [np.nanmin(Var_inter_mat_mean), np.nanmax(Var_inter_mat_mean)], 'k', lw=0.5)

if np.any(NFI_ind == np.array([4, 5, 6])):
	ax3.set_xlabel("DTM mean elevation by SER", fontsize=fs-2, labelpad=-0.25)

ax3.set_ylabel("Between variance by SER", fontsize=fs-2)
ax3.yaxis.set_label_position("right")

im3 = ax3.scatter(MNT_elev_mat_mean, Var_inter_mat_mean, c='k', marker="o", s=3)

ax3.set_aspect((np.nanmax(MNT_elev_mat_mean)-np.nanmin(MNT_elev_mat_mean))/(np.nanmax(Var_inter_mat_mean)-np.nanmin(Var_inter_mat_mean)))

ax3.set_xticks([])
ax3.set_yticks([])

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

ax3.set_title("Pearson correlation r = %.2f" % Corr_MNT_elev_SER_inter[0,1], fontsize = fs-2)

ax4 = plt.subplot2grid((1, 4), (0, 3))

im4 = ax4.plot([np.nanmin(MNT_pente_mat_mean), np.nanmax(MNT_pente_mat_mean)], [np.nanmin(Var_inter_mat_mean), np.nanmax(Var_inter_mat_mean)], 'k', lw=0.5)

if np.any(NFI_ind == np.array([4, 5, 6])):
	ax4.set_xlabel("DTM mean slope by SER", fontsize=fs-2, labelpad=-0.25)

ax4.set_ylabel("Between variance by SER", fontsize=fs-2)
ax4.yaxis.set_label_position("right")

im4 = ax4.scatter(MNT_pente_mat_mean, Var_inter_mat_mean, c='k', marker="o", s=3)

ax4.set_aspect((np.nanmax(MNT_pente_mat_mean)-np.nanmin(MNT_pente_mat_mean))/(np.nanmax(Var_inter_mat_mean)-np.nanmin(Var_inter_mat_mean)))

ax4.set_xticks([])
ax4.set_yticks([])

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)

ax4.set_title("Pearson correlation r = %.2f" % Corr_MNT_pente_SER_inter[0,1], fontsize = fs-2)

plt.tight_layout(pad = 2)

plt.savefig("GMD_illustrations/Variances_corr_"+str+"_"+str_dist+".png", dpi=my_dpi)