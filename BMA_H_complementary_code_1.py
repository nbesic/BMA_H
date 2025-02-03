#	N. Besic, N. Picard, C. Vega, J.-D. Bontemps, L. Hertzog, J.-P. Renaud, 
#	F. Fogel, M. Schwartz, A. Pellissier-Tanon, G. Destouet, F. Mortier, 
#	M. Planells-Rodriguez, and P. Ciais , “Remote sensing-based forest canopy 
# 	height mapping: some models are useful, but might they provide us with even 
#	more insights when combined?,” Geoscientific Model Development,
#	18, 337–359, 2025. DOI: 10.5194/gmd-18-337-2025.
#
#	Complementary code 1 (Fig. 2)
#	Please see Main code (Fig. 3, 4, 5, 7 and B1)
#	Please see complementary code 2 (Fig. 6 and A1)
#
#	Fig. 1 and A2 represent respectively a diagram and 
#	an illustration and are established using TikZ and QGIS 
#	and are therefore not reproductable with the python code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

my_dpi = 500

fs = 9

pdf_ind = 1 # 1- normal, 2 - lognormal

if pdf_ind == 1:
	str_dist = "norm"
elif pdf_ind == 2:
	str_dist = "log_norm"

Weights = pd.DataFrame()

Weights["M"] = np.array(["$M_1$", "$M_2$", "$M_3$", "$M_4$", "$M_5$"])


for NFI_ind in range(1,8):

	if NFI_ind == 1:
		str = "h_max"
	elif NFI_ind == 2:
		str = "h_mean"
	elif NFI_ind == 3:
		str = "h_dom"
	elif NFI_ind == 4:
		str = "hi_max"
	elif NFI_ind == 5:
		str = "hi_mean"
	elif NFI_ind == 6:
		str = "hi_dom"
	elif NFI_ind == 7:
		str = "hi_lor"
		
	w = np.load("GMD_data_output\Weights_"+str+"_"+str_dist+".npy")
	
	Weights[str] = w
	
Weights["h_lor"] = np.zeros(5)
	
print(Weights)

cmap_colors = np.array([[0.23192618223760095, 0.5456516724336793, 0.7626143790849673, 1.0],
						[0.48402921953094963, 0.4543329488658208, 0.7106651287966167, 1.0],
					    [0.907912341407151, 0.20284505959246443, 0.16032295271049596, 1.0],
					    [0.9255363321799308, 0.3848673587081891, 0.05983852364475202, 1.0],
						[0.23044982698961936, 0.6445059592464436, 0.34514417531718566, 1.0]])

custom_cmap= ListedColormap(cmap_colors)

## Illustration - overall weights (Fig. 2 in the manuscript)

fig = plt.figure(figsize=(15,6))

ax1 = plt.subplot2grid((1, 4), (0, 0))

ax1.bar(Weights["M"], Weights["h_dom"], width= -0.4, align = "edge", color = cmap_colors, hatch="")
ax1.bar(Weights["M"], Weights["hi_dom"], width= 0.4, align = "edge", color = cmap_colors, hatch="////")
ax1.set_title("Dominant height - $H_{\mathrm{dom}}$")
ax1.set_xlabel("Model")
ax1.set_ylabel("BMA weights")
ax1.set_ylim(0,0.4)
ax1.set_aspect(len(Weights["M"])*2)

ax1.grid(color = "gray", linestyle = "dashed", axis = "y")

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

handles = [
    Patch(facecolor="#ffffff", hatch="", label="WITHOUT imputations"),
    Patch(facecolor="#ffffff", hatch="////", label="WITH imputations")
]

ax1.legend(handles = handles, loc = 2)

ax2 = plt.subplot2grid((1, 4), (0, 1))

ax2.bar(Weights["M"], Weights["h_mean"], width= -0.4, align = "edge", color = cmap_colors, hatch="")
ax2.bar(Weights["M"], Weights["hi_mean"], width= 0.4, align = "edge", color = cmap_colors, hatch="////")
ax2.set_title("Mean height - $H_{\mathrm{mean}}$")
ax2.set_xlabel("Model")
ax2.set_ylim(0,0.4)
ax2.set_aspect(len(Weights["M"])*2)

ax2.grid(color = "gray", linestyle = "dashed", axis = "y")

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax3 = plt.subplot2grid((1, 4), (0, 2))

ax3.bar(Weights["M"], Weights["h_max"], width= -0.4, align = "edge", color = cmap_colors, hatch="")
ax3.bar(Weights["M"], Weights["hi_max"], width= 0.4, align = "edge", color = cmap_colors, hatch="////")
ax3.set_title("Maximum height - $H_{\mathrm{max}}$")
ax3.set_xlabel("Model")
ax3.set_ylim(0,0.4)
ax3.set_aspect(len(Weights["M"])*2)

ax3.grid(color = "gray", linestyle = "dashed", axis = "y")

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

ax4 = plt.subplot2grid((1, 4), (0, 3))

ax4.bar(Weights["M"], Weights["h_lor"], width= -0.4, align = "edge", color = cmap_colors, hatch="")
ax4.bar(Weights["M"], Weights["hi_lor"], width= 0.4, align = "edge", color = cmap_colors, hatch="////")
ax4.set_title("Lorey's height - $H_{\mathrm{lor}}$")
ax4.set_xlabel("Model")
ax4.set_ylim(0,0.4)
ax4.set_aspect(len(Weights["M"])*2)

ax4.grid(color = "gray", linestyle = "dashed", axis = "y")

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)

plt.savefig("GMD_illustrations/Weights_bar_plot.png", dpi=my_dpi)
