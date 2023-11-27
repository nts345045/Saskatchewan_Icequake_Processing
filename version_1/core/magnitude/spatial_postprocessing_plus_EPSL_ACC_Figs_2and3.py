"""
spatial_postprocessing.py

This takes outputs from postprocess_M0_estimates.py and the bed topography model
and estimates the following metrics:

For each event:

For each model gridpoint:
 - Average bedslope as mEm and mNm basis vectors in a user-specified radius
 - binned statistics for each model point using a user-specified radius for binning

Results are output as grid files readable in QGIS

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from tqdm import tqdm
from pyproj import Proj

# Define root directory path
ROOT = os.path.join("..", "..")
# BedMap Directory
BMDR = os.path.join(ROOT, "data", "seismic", "HVSR", "grids")
# Get filtered moment estimates
M0FL = os.path.join(
    ROOT, "data", "M0_results", "M0_stats_SNR_ge5_ND_ge5_MA_le3_SA_wt.csv"
)
# Map moulin locations
MOUL = os.path.join(ROOT, "data", "tables", "Moulin_Locations.csv")
# Map station locations
# Station locations
STFL = os.path.join(ROOT, "data", "seismic", "Stations.csv")
# Figure output directory
ODIR = os.path.join(ROOT, "outputs")

issave = True
DPI = 300
FMT = "PNG"

## USER DEFINED PARAMETERS ##
RSAMP_M0 = 10  # [m] Sampling radius for moment release calculations (scale to event location uncertainties)
RSAMP_SLP = 80  # [m] Sampling radius for determining slope (scale to station spacing)
NAN_VAL = np.nan  # [-] Value to use for NaN Gridpoints
DX = 5.0  # [m] X-axis (East) sampling interval for grd data
DY = 5.0  # [m] Y-axis (North) sampling interval for grd data
FD_AZ = 44.5  # [N deg E] Flow direction Azimuth


def write_ascii_raster(file, EE, NN, ZZ, nodata_value=-9999):
    # Open File, appending extension if needed
    if file.split(".")[-1] != "ascii":
        fobj = open(file + ".ascii", "w")
    else:
        fobj = open(file, "w")
    # Write Header
    fobj.write("ncols %d\n" % (EE.shape[1]))
    fobj.write("nrows %d\n" % (EE.shape[0]))
    fobj.write("xllcorner %.3f\n" % (np.nanmin(EE)))
    fobj.write("yllcorner %.3f\n" % (np.nanmin(NN)))
    fobj.write("cellsize %.3f\n" % (np.nanmean(EE[:, 1:] - EE[:, :-1])))
    fobj.write("nodata_value %.3f\n" % (nodata_value))

    # Iterate across input griddata
    for i_ in np.arange(EE.shape[0], 0, -1) - 1:
        for j_ in range(EE.shape[1]):
            z_ij = ZZ[i_, j_]
            # Check if nodata (handles NaN & Inf)
            if not np.isfinite(z_ij):
                z_ij = nodata_value
            # Write entry ij
            fobj.write("%.3f" % (z_ij))
            # If not done with the line, add a space
            if j_ + 1 < EE.shape[1]:
                fobj.write(" ")
            # If done with the line, but more lines are present, add a return
            elif i_ > 0:
                fobj.write("\n")
    # Finish Writing
    fobj.close()


## Import Grids
grd_E = np.load(os.path.join(BMDR, "mE_Grid.npy"))
grd_N = np.load(os.path.join(BMDR, "mN_Grid.npy"))
grd_Hm = np.load(os.path.join(BMDR, "H_med_Grid.npy"))
grd_Hu = np.load(os.path.join(BMDR, "H_gau_u_Grid.npy"))
grd_Ho = np.load(os.path.join(BMDR, "H_gau_o_Grid.npy"))
grd_Zs = np.load(os.path.join(BMDR, "mZsurf_Grid.npy"))
grd_M = np.load(os.path.join(BMDR, "MASK_Grid.npy"))

# Load points
df_STA = pd.read_csv(STFL)
df_MOU = pd.read_csv(MOUL)

## Convert Station Locations to UTM
myproj = Proj(proj="utm", zone=11, ellipse="WGS84", preserve_units=False)
SmE, SmN = myproj(df_STA["Lon"].values, df_STA["Lat"].values)
df_STA = pd.concat(
    [df_STA, pd.DataFrame({"mE": SmE, "mN": SmN}, index=df_STA.index)],
    axis=1,
    ignore_index=False,
)

df_STA = df_STA[~df_STA["Station"].isin(["1129", "1130"])]

SmE, SmN = myproj(df_MOU["lon"].values, df_MOU["lat"].values)
df_MOU = pd.concat(
    [df_MOU, pd.DataFrame({"mE": SmE, "mN": SmN}, index=df_MOU.index)],
    axis=1,
    ignore_index=False,
)

df_MOU = df_MOU[
    (df_MOU["mE"] <= grd_E.max())
    & (df_MOU["mE"] >= grd_E.min())
    & (df_MOU["mN"] <= grd_N.max())
    & (df_MOU["mN"] >= grd_N.min())
]

# Load Filtered M0 event estimates
df_M = pd.read_csv(M0FL, parse_dates=["t0"], index_col="t0")
df_M = df_M.sort_index()

# Calculate Individual point gradients
grd_mEm, grd_mNm = np.gradient((grd_Zs - grd_M) * grd_M, DX, DY)

grd_med_M0 = []
grd_e_count = []
grd_mEm_avg = []
grd_mNm_avg = []
grd_max_M0 = []

for I_, E_ in enumerate(grd_E[0, :]):
    med_M0_line = []
    max_M0_line = []
    ct_line = []
    mEm_avg_line = []
    mNm_avg_line = []

    for J_, N_ in enumerate(grd_N[:, 0]):
        # Subset events
        idf_M = df_M[
            np.sqrt((df_M["mE"].values - E_) ** 2 + (df_M["mN"].values - N_) ** 2)
            <= RSAMP_M0
        ]
        # Grab event count (for event density grid)
        ct_line.append(len(idf_M))
        if len(idf_M) > 0:
            # Calculate median M0
            med_M0_line.append(np.nanmedian(idf_M["SA_wt_mean"]))
            max_M0_line.append(np.nanmax(idf_M["SA_wt_mean"]))
        else:
            med_M0_line.append(NAN_VAL)
            max_M0_line.append(NAN_VAL)

        # Calculate distance matrix for grids
        DDg = np.sqrt((grd_E - E_) ** 2 + (grd_N - N_) ** 2)

        # Get aerally averaged estimates of slope elements
        mEm_avg_line.append(np.nanmedian(grd_mEm[DDg <= RSAMP_SLP]))
        mNm_avg_line.append(np.nanmedian(grd_mNm[DDg <= RSAMP_SLP]))

    # Place lines into arrays
    grd_med_M0.append(med_M0_line)
    grd_max_M0.append(max_M0_line)
    grd_e_count.append(ct_line)
    grd_mEm_avg.append(mEm_avg_line)
    grd_mNm_avg.append(mNm_avg_line)


# Convert nested lists into numpy arrays
grd_med_M0 = np.array(grd_med_M0)
grd_max_M0 = np.array(grd_max_M0)
grd_e_count = np.array(grd_e_count)
grd_mEm_avg = np.array(grd_mEm_avg)
grd_mNm_avg = np.array(grd_mNm_avg)


# Calculate along-flow slope coefficients
fdE = np.cos((np.pi / 180.0) * (90.0 - FD_AZ))
fdN = np.sin((np.pi / 180.0) * (90.0 - FD_AZ))


## Get slope estimates for each event
holder = []
for K_ in tqdm(range(len(df_M))):
    # Get event Series
    KS_ = df_M.iloc[K_, :]
    # Calculate distance matrix
    DDe = np.sqrt((grd_E - KS_["mE"]) ** 2 + (grd_N - KS_["mN"]) ** 2)
    # Get indices of the minimum distance point
    xm = np.argmin(np.abs(grd_E[0, :] - KS_["mE"]))
    ym = np.argmin(np.abs(grd_N[:, 0] - KS_["mN"]))

    mEmK = float(grd_mEm_avg[xm, ym])
    mNmK = float(grd_mNm_avg[xm, ym])
    line = [mEmK, mNmK, np.min(DDe), int(xm), int(ym)]
    holder.append(line)

df_Md = pd.concat(
    [
        df_M,
        pd.DataFrame(
            holder, columns=["mEm", "mNm", "NND m", "indX", "indY"], index=df_M.index
        ),
    ],
    axis=1,
    ignore_index=False,
)


# Calculate along-flow slope grid
grd_Mdf_avg = 90 * (fdE * grd_mEm_avg + fdN * grd_mNm_avg) * grd_M.T
# Write slope grid to ASCII raster file
write_ascii_raster(
    os.path.join(ODIR, "Along_flow_slopes_deg_masked"),
    grd_E,
    grd_N,
    grd_Mdf_avg.T,
    nodata_value=-9999,
)
# Write median M0 to ASCII raster file
write_ascii_raster(
    os.path.join(ODIR, "M0_median_12p5m_radius"),
    grd_E,
    grd_N,
    grd_med_M0.T * grd_M,
    nodata_value=-9999,
)
# Write median mw to ASCII raster file
write_ascii_raster(
    os.path.join(ODIR, "mw_median_5m_radius"),
    grd_E,
    grd_N,
    ((2 / 3) * np.log10(grd_med_M0.T) - 6.07) * grd_M,
    nodata_value=-9999,
)
# Write median mw to ASCII raster file
write_ascii_raster(
    os.path.join(ODIR, "mw_max_5m_radius"),
    grd_E,
    grd_N,
    ((2 / 3) * np.log10(grd_max_M0.T) - 6.07) * grd_M,
    nodata_value=-9999,
)

# Grab data for event shown in Fig. S3
FigS3_M = df_Md[df_Md["EVID"] == 21414]


## GENERATE FIGURE 2 FORM MAIN TEXT ##
fig = plt.figure(figsize=(6.1, 9.5))
gs = fig.add_gridspec(nrows=2, ncols=1, wspace=0, hspace=0)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])

# Place topography shading
axA.pcolor(grd_E, grd_N - 5.777e6, grd_M * (grd_Zs - grd_Hu), cmap="Greys_r")

# Plot event density & colorbar
ch = axA.pcolor(grd_E, grd_N - 5.777e6, np.log10(grd_e_count.T), cmap="inferno", vmax=3)
cbar = plt.colorbar(ch, ax=axA, location="right", orientation="vertical", shrink=0.5)
cbar.set_ticks([0, 1, 2, 3])
cbar.set_ticklabels([1, 10, 100, 1000])
cbar.set_label("Event density (ct.)", rotation=270, labelpad=15)
# Plot topography contours
cl = axA.contour(
    grd_E,
    grd_N - 5.777e6,
    grd_M * (grd_Zs - grd_Hu),
    levels=np.arange(1010, 2000, 10),
    colors="w",
    linewidths=0.5,
    zorder=5,
)
axA.clabel(cl, levels=cl.levels[::2], fmt="%d", fontsize=8, colors="k", zorder=2)
for label in cl.labelTexts:
    label.set_path_effects(
        [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
    )


# axA.axis('square')
axA.xaxis.set_ticks(np.arange(487750, 488500, 250))
axA.set_xlim([df_M["mE"].min() - 200, df_M["mE"].max()])

# axA.plot(FigS3_M['mE'],FigS3_M['mN'] - 5.777e6,'*',color='red',ms=9,label='Fig. 2 icequake',zorder=1)
# Plot Moulins
axA.plot(
    df_MOU["mE"],
    df_MOU["mN"] - 5.777e6,
    "*",
    color="cyan",
    ms=9,
    label="Moulins",
    zorder=1,
)
axA.legend(loc="lower right")

# Place topography shading
axB.pcolor(grd_E, grd_N - 5.777e6, grd_M * grd_Mdf_avg.T, cmap="Greys_r")

# Plot maximum local density & colorbar
ch = axB.pcolor(
    grd_E, grd_N - 5.777e6, (2 / 3) * np.log10(grd_max_M0.T) - 6.03, cmap="inferno"
)
cbar = plt.colorbar(ch, ax=axB, location="right", orientation="vertical", shrink=0.5)
# cbar.set_ticks([0,1,2,3])
# cbar.set_ticklabels([1,10,100,1000])
cbar.set_label("Peak $m_w$", rotation=270, labelpad=15)
# Plot topography contours
cl = axB.contour(
    grd_E,
    grd_N - 5.777e6,
    grd_M * grd_Mdf_avg.T,
    levels=np.arange(-25, 30, 5),
    colors="w",
    linewidths=0.75,
)
axB.clabel(cl, levels=cl.levels[::1], fmt="%.1f$^o$", fontsize=8, colors="k")
for label in cl.labelTexts:
    label.set_path_effects(
        [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
    )
# Plot Stations
axB.plot(df_STA["mE"], df_STA["mN"] - 5.777e6, "cv", ms=4, label="Geophones")
axB.legend(loc="lower right")

# axB.axis('square')
axB.xaxis.set_ticks(np.arange(487750, 488500, 250))
axB.set_xlim([df_M["mE"].min() - 200, df_M["mE"].max()])


axA.axis("square")
axB.axis("square")

axA.set_xlim((487540.99248120305, 488255.99248120305))
axB.set_xlim((487540.99248120305, 488255.99248120305))
axA.set_ylabel("Northing UTM 11N (+ %d m)" % (5.777e6))
axB.set_ylabel("Northing UTM 11N (+ %d m)" % (5.777e6))
axA.xaxis.set_visible(False)
axB.set_xlabel("Easting UTM 11N (m)")
for ax, ll in [(axA, "A"), (axB, "B")]:
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.text(
        xlims[0] + 0.05 * (xlims[1] - xlims[0]),
        ylims[1] - 0.05 * (ylims[1] - ylims[0]),
        ll,
        fontweight="extra bold",
        fontstyle="italic",
        fontsize=14,
        ha="center",
        va="center",
    )

if issave:
    I_OFILE = "EPSL_ACC_Fig2_Event_Density_Peak_mw_Maps_%ddpi.%s" % (DPI, FMT.lower())
    plt.savefig(os.path.join(ODIR, I_OFILE), dpi=DPI, format=FMT)
    print("Figure saved: %s" % (os.path.join(ODIR, I_OFILE)))


## GENERATE FIGURE 3 FOR MAIN TEXT ##
fig = plt.figure(figsize=(9.8, 7))
gs = fig.add_gridspec(nrows=5, ncols=5, wspace=0, hspace=0)
axA = fig.add_subplot(gs[:-1, :-1])
axC = fig.add_subplot(gs[-1, 1:-1])
axB = fig.add_subplot(gs[:-1, -1])
# xd = np.log10(grd_med_M0).ravel()
# yd = (180/np.pi)*np.pi*0.5*grd_Mdf_avg.ravel()# - np.nanmedian(grd_Mdf_avg)
idf_Md = df_Md[df_Md["SA_wt_mean"].notna()]
xd = (2 / 3) * np.log10(idf_Md["SA_wt_mean"]) - 6.03
yd = 90 * (fdE * idf_Md["mEm"] + fdN * idf_Md["mNm"])
outs = axA.hist2d(xd, yd, bins=100, cmin=1, cmap="inferno", zorder=2)
cbar = plt.colorbar(
    outs[3], ax=axA, location="left", orientation="vertical", shrink=0.5
)
cbar.set_label("Icequake counts")
cbar.set_ticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90])
cbar.set_ticklabels([1, 10, 20, 30, 40, 50, 60, 70, 80, 90])
# axA.plot((2/3)*np.log10(FigS3_M['SA_wt_mean']) - 6.03,90*(fdE*FigS3_M['mEm'] + fdN*FigS3_M['mNm']),\
# 		 '*',ms=9,color='red',label='Fig. 2 icequake')
# axA.legend()


houts = axC.hist(xd, 100, log=True, zorder=1)
# Do Pareato distribution fit on the limited range for Gutenberg & Richter (1955) model fitting
mw_min = -2.7
mw_max = -1.5
ofx = (houts[1][1:] + houts[1][:-1]) / 2
ofy = np.log10(houts[0])
IND = (ofx >= mw_min) & (np.isfinite(ofy)) & (ofx <= mw_max)

axC.hist(xd[xd > mw_max], bins=houts[1])

mod, cov = np.polyfit(ofx[IND], ofy[IND], 1, cov=True)
xhatv = np.linspace(mw_min, -1, 101)
axC.semilogy(xhatv, 10 ** (np.poly1d(mod)(xhatv)), "r-.", lw=2)
axC.text(
    -1.3,
    1e2,
    "G-R model fit\nb: %.2f$\\pm$%.2f\na: %.2f$\\pm$%.2f"
    % (-1 * mod[0], cov[0, 0] ** 0.5, mod[1], cov[1, 1] ** 0.5),
    color="red",
    va="center",
)


# Histogram of event-slope frequency
h1 = axB.hist(
    yd,
    80,
    orientation="horizontal",
    density=True,
    zorder=1,
    alpha=0.5,
    label="Localized",
)
# Histogram of all-bed frequency
h2 = axB.hist(
    grd_Mdf_avg.ravel(),
    h1[1],
    orientation="horizontal",
    density=True,
    zorder=1,
    alpha=0.5,
    label="Imaged Area",
)

# h3 = axB.plot((h1[0]/h2[0])*(h1[0].max()/40),(h1[1][1:] + h1[1][:-1])/2,'k-',label='Normalized')

axB.legend()
# axA.plot(xd[xd > -2],yd[xd > -2],'rs',ms=0.5)
# axA.plot(xd[xd <-3.5],yd[xd <-3.5],'c*',ms=0.5)
# axA.plot(xd[(yd <-16)&(xd <= -2)&(xd >=-3.5)],yd[(yd <-16)&(xd <= -2)&(xd >=-3.5)],'w.',ms=1)

# Scaling adjustments
axC.set_xlim(-4.25, 0.25)
axB.set_ylim(17.5, -27.25)

# Overlay average bedslope
axA.plot(
    axC.get_xlim(),
    np.ones(
        2,
    )
    * np.nanmean(grd_Mdf_avg),
    "r:",
    alpha=0.9,
    lw=2,
    zorder=3,
)
axB.plot(
    axB.get_xlim(),
    np.ones(
        2,
    )
    * np.nanmean(grd_Mdf_avg),
    "r:",
    alpha=0.9,
    lw=2,
    zorder=2,
)
axA.text(
    -0.4,
    np.nanmean(grd_Mdf_avg),
    "Average\nslope",
    color="r",
    ha="center",
    va="center",
    fontweight="bold",
    fontsize=13,
)

axA.set_ylim(axB.get_ylim())
axA.set_xlim(axC.get_xlim())

# Axis label/tick formatting
axA.xaxis.set_visible(True)
axA.xaxis.set_ticks_position("top")
axA.xaxis.set_label_position("top")
axB.yaxis.set_ticks_position("right")
axB.yaxis.set_label_position("right")
axA.yaxis.set_visible(True)

# Subplot labels
axA.text(
    -4,
    -25,
    "A",
    fontsize=16,
    fontweight="extra bold",
    fontstyle="italic",
    ha="center",
    va="center",
)
axC.text(
    -4,
    10**2.5,
    "C",
    fontsize=16,
    fontweight="extra bold",
    fontstyle="italic",
    ha="center",
    va="center",
)
axB.text(
    0.20,
    15,
    "B",
    fontsize=16,
    fontweight="extra bold",
    fontstyle="italic",
    ha="right",
    va="center",
)
axB.set_xlim([0.001, 0.21])
# Axis labels
axA.set_xlabel("Moment magnitude [$m_w$]")
axC.set_xlabel("Moment magnitude [$m_w$]")
axC.set_ylabel("Counts")
axB.set_xlabel("Frequency")
axB.set_ylabel("Along-flow bed slope [$^o$ from horiontal]", rotation=270, labelpad=15)

# Annotate slope types
axA.text(-2, -20, "Prograde bed", ha="center", va="center")
axA.text(-2, 16, "Retrograde bed", ha="center", va="center")


if issave:
    I_OFILE = "EPSL_ACC_Fig3_Bed_Geometry_Magnitude_Relationships_%ddpi.%s" % (
        DPI,
        FMT.lower(),
    )
    plt.savefig(os.path.join(ODIR, I_OFILE), dpi=DPI, format=FMT)
    print("Figure saved: %s" % (os.path.join(ODIR, I_OFILE)))


plt.show()


# # Transforme some data for plotting
# v_med_M0 = grd_med_M0.ravel()


# plt.figure()
# plt.subplot(221)
# plt.pcolor(grd_E,grd_N,(grd_Zs - grd_Hu)*grd_M)
# plt.xlabel('Easting')
# plt.ylabel('Northing')
# plt.colorbar()
# plt.title('Bed Elevation')
# plt.axis('square')


# plt.subplot(222)
# zarg = grd_mEm_avg.T*grd_M
# plt.pcolor(grd_E,grd_N,zarg - np.nanmedian(zarg))
# plt.colorbar()
# plt.contour(grd_E,grd_N,zarg - np.nanmedian(zarg),[0])
# plt.contour(grd_E,grd_N,zarg - np.nanmedian(zarg),np.linspace(-0.4,0.4,17))
# plt.xlabel('Easting')
# plt.ylabel('Northing')
# plt.title('Easting slope')
# plt.axis('square')

# plt.subplot(223)
# zarg = grd_mNm_avg.T*grd_M
# plt.pcolor(grd_E,grd_N,zarg - np.nanmedian(zarg))
# plt.colorbar()
# plt.contour(grd_E,grd_N,zarg - np.nanmedian(zarg),[0])
# plt.contour(grd_E,grd_N,zarg - np.nanmedian(zarg),np.linspace(-0.4,0.4,17))
# plt.xlabel('Easting')
# plt.ylabel('Northing')
# plt.title('Northing slope')
# plt.axis('square')

# plt.subplot(224)
# zarg = grd_Mdf_avg.T*grd_M
# plt.pcolor(grd_E,grd_N,zarg - np.nanmedian(zarg))
# plt.colorbar()
# plt.contour(grd_E,grd_N,zarg - np.nanmedian(zarg),[0])
# plt.contour(grd_E,grd_N,zarg - np.nanmedian(zarg),np.linspace(-0.4,0.4,17))
# plt.xlabel('Easting')
# plt.ylabel('Northing')
# plt.title('Along-flow slope')
# plt.axis('square')
# plt.plot(df_M['mE'],df_M['mN'],'r.',alpha=0.01)


# plt.figure()
# # Plot just event count histogram
# plt.subplot(221)
# plt.hist((grd_Mdf_avg*(grd_e_count/grd_e_count)).ravel(),100)
# plt.xlabel('Prograde      Bedslope [m/m]       Retrograde')
# plt.plot(np.ones(2,)*np.nanmedian(grd_Mdf_avg.T*grd_M),plt.gca().get_ylim(),'r:')

# # plot scatter plot of log10(M0) and along-flow bedslope
# plt.subplot(222)
# plt.plot(np.log10(grd_med_M0.ravel()),grd_Mdf_avg.ravel() - np.nanmedian(grd_Mdf_avg),'.',alpha=0.1)


# plt.subplot(223)
