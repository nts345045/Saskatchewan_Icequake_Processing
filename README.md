# Saskatchewan_Icequake_Processing
Alpine glacier seismicity analysis codes custom-developed for the dense nodal array deployment at Saskatchewan Glacier, Canadian Rocky Mountains

EPSL_Accepted_Codebase/ contains archived processing scripts from analyses in 2019/2020 used to develop the basal ice-quake catalog presented in a number of talks by N.T. Stevens and others from the AUG 2019 nodal seismic deployment on Saskatchewan Glacier and the dissertation by N.T. Stevens.

version_2/ contains processing scripts that expand upon version_1/ including a parallelized implementation of the F-statistic dynamic threshold from Carmichael et al. (2013, 2015) and updated use of parallelization libraries.

These codes heavily leverage SQLite database interfaces and schema provided by the PISCES project (https://github.com/LANL-Seismoacoustics/pisces) from its earliet release on GitHub. In future versions (version_3 | new MAIN branch) these dependencies may be relaxed in favor of use of Pandas for relational
table management to mitigate obsolescence and use of more-popular tabular data interfaces.

Waveform and initial catalog data are archived at: https://minds.wisconsin.edu/handle/1793/83756 (and links to the IRIS-DMC therein).

If you find these codes useful, please cite:

Stevens, N.T. (2022) "Constraints on transient glacier slip with ice-bed separation" Doctoral Dissertation, University of Wisconsin - Madison. ISBN:9798841732495 

(peer-reviewed article forthcoming)



REFERENCES:

Carmichael, J.D., 2013. Melt-Triggered Seismic Response in Hydraulically-Active Polar Ice: Observations and Methods. University of Washington.

Carmichael, J.D., Joughin, I., Behn, M.D., Das, S., King, M.A., Stevens, L., Lizarralde, D., 2015. Seismicity on the western Greenland Ice Sheet: Surface fracture in the vicinity of active moulins. Journal of Geophysical Research: Earth Surface 120, 1082–1106. https://doi.org/10.1002/2014JF003398

McBrearty, I.W., Zoet, L., Anandakrishnan, S., 2020. Basal seismicity of the Northeast Greenland Ice Stream. Journal of Glaciology 1.


