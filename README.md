# Energy-Based Scatter Estimation in PET

This repository contains the simulation and analysis code used for the MSc research project on energy-based scatter estimation (EBSE) in positron emission tomography (PET).

The project investigates the estimation of scattered and unscattered coincidence events from the measured two-dimensional photon-energy distribution. Monte Carlo data generated using SimSET are used to evaluate different methods for estimating the energy probability density functions (PDFs), constructing energy basis functions, and estimating their spatial coefficients using NEGML.

The main analysis is carried out in `Procedure.ipynb`, with the underlying calculations separated into Python modules according to their role in the analysis.

---

## Analysis workflow

The main analysis follows the sequence

SimSET energy-resolved projection data  
→ energy-resolved \(C_0,C_1,C_2,C_3\) components  
→ non-scatter and scatter energy PDFs  
→ 2D energy basis functions  
→ NEGML coefficient estimation  
→ estimated scatter and non-scatter sinograms  
→ comparison with Monte Carlo ground truth.

The four simulated coincidence components are:

- \(C_0\): neither detected photon scattered;
- \(C_1\): first photon unscattered, second photon scattered;
- \(C_2\): first photon scattered, second photon unscattered;
- \(C_3\): both detected photons scattered.

The total non-scatter estimate is therefore associated with \(C_0\), while the total scatter estimate is obtained from \(C_1+C_2+C_3\).

---

## Repository structure

### `Procedure.ipynb`

Main analysis notebook.

The notebook loads the simulated datasets and runs the different EBSE experiments, including:

- global and spatially varying basis models;
- predicted and Monte Carlo ground-truth energy PDFs;
- shared and separate 1D factorised basis functions;
- direct ground-truth 2D basis functions;
- axial averaging;
- different spatial region sizes;
- representative-pixel diagnostics;
- investigations of detector energy resolution and count statistics.

The notebook is intended to provide the high-level experimental workflow, while most reusable calculations are implemented in the Python modules described below.

### `data_loading.py`

Functions for reading the energy-resolved SimSET/STIR projection data.

The SimSET output is separated into files of the form

`blue0_en[i]_pink0_en[j]`  
`blue0_en[i]_pink1_en[j]`  
`blue1_en[i]_pink0_en[j]`  
`blue1_en[i]_pink1_en[j]`

where `i` and `j` identify the energy bins of the two detected photons and the blue/pink indices describe their simulated scatter histories.

The module constructs the energy-resolved \(C_0\)-\(C_3\) arrays and their sum, as well as energy-integrated ground-truth scatter and non-scatter sinograms.

### `energy_pdfs.py`

Functions for constructing one-dimensional photon-energy PDFs.

This includes:

- derivation of Monte Carlo ground-truth non-scatter and scatter PDFs;
- prediction of the non-scatter PDF using a Gaussian detector-energy response centred at 511 keV;
- prediction of the scatter PDF by scaling and subtracting the estimated non-scatter contribution from the total marginal energy spectrum.

The energy PDFs are numerically normalised over the investigated energy range before being used to construct the basis functions.

Parts of the PDF-estimation procedure were adapted from previous EBSE work and reorganised for the analysis used in this project. See the source file for attribution details.

### `basis_functions.py`

Construction of the four 2D energy basis functions used by NEGML.

Three basis models are implemented.

**Shared 1D factorised basis**

The same non-scatter and scatter PDFs are used for both photon-energy dimensions:

\(U(E_1)U(E_2)\),  
\(U(E_1)S(E_2)\),  
\(S(E_1)U(E_2)\),  
\(S(E_1)S(E_2)\).

**Separate 1D factorised basis**

Separate marginal PDFs are retained for the two photon-energy dimensions:

\(U_1(E_1)U_2(E_2)\),  
\(U_1(E_1)S_2(E_2)\),  
\(S_1(E_1)U_2(E_2)\),  
\(S_1(E_1)S_2(E_2)\).

**Direct ground-truth 2D basis**

The four Monte Carlo \(C_0,C_1,C_2,C_3\) joint-energy histograms are normalised directly. This avoids the assumption that the 2D distributions can be factorised into products of 1D marginal PDFs and is used as a diagnostic reference.

### `negml.py`

Implementation of the NEGML coefficient-estimation procedure.

The module contains:

- construction of the 1D system matrix;
- extraction of the two marginal prompt-energy spectra;
- covariance estimation and initial coefficient estimation;
- the four-parameter NEGML update;
- application of NEGML across the sinogram.

The four fitted coefficients correspond to the \(UU\), \(US\), \(SU\), and \(SS\) basis components.

### `spatial_regions.py`

Utilities for extracting spatially restricted energy data.

The functions in this module are used to:

- construct spatial masks;
- extract joint energy histograms within a selected region;
- derive ground-truth \(U_1,S_1,U_2,S_2\) PDFs within a region;
- extract the individual \(C_0-C_3\) components for representative-pixel analysis.

These functions support the comparison between a single global basis and spatially varying basis functions.

### `estimation_workflows.py`

Higher-level workflows that combine PDF estimation, basis construction and NEGML fitting.

This module includes procedures for:

- fitting all axial sinogram positions independently;
- averaging the energy-resolved data along the axial direction before fitting;
- applying a basis to rectangular spatial regions;
- applying the different PDF and basis configurations to individual representative pixels.

The main notebook calls these workflows rather than repeating the full estimation procedure for every experiment.

### `Graphic_display.py`

Plotting and result-display utilities.

This includes functions for separating the four fitted NEGML coefficients, automatically generating labels for different PDF/basis configurations, comparing estimated and ground-truth sinograms, and displaying individual 2D basis functions.

---

## Simulation data

The projection data were generated using SimSET and converted to STIR-compatible energy-resolved sinograms.

The reference dataset used for the main analysis has:

- detector energy resolution: 10% FWHM at 511 keV;
- simulated decays: \(10^9\);
- 10 energy bins for each detected photon;
- energy range: 400--600 keV;
- energy-bin width: 20 keV;
- sinogram view dimension: 32;
- tangential dimension: 55;
- axial bins: 12.

Additional simulations were generated with different detector energy resolutions and count statistics for sensitivity studies.

The SimSET parameter files define the detector model, energy and sinogram binning, activity distribution, attenuation distribution and photon-history output. The detailed simulation settings are described in the dissertation.

---

## Spatial basis estimation

The analysis supports both global and spatially varying energy basis functions.

For the global model, one set of energy PDFs and basis functions is derived from the complete view--tangential sinogram plane.

For the multi-region model, the \(32\times55\) view--tangential plane is divided into rectangular regions. Separate energy information is derived for each region before NEGML is applied to the pixels within that region.

The investigated region widths are

`[1, 2, 4, 8, 16, 32]`

and the investigated region lengths are

`[1, 5, 11, 55]`.

A \(32\times55\) region therefore corresponds to the global case, whereas a \(1\times1\) region provides a separate local basis for each view--tangential pixel.

---

## Requirements

The analysis uses Python together with the following main packages:

- NumPy
- SciPy
- Matplotlib
- SIRF/STIR

SimSET is required to reproduce the Monte Carlo simulations.

Exact software installation and configuration are not reproduced in this repository README. The dissertation provides further information and refers to the previous project setup where appropriate.

---

## Running the analysis

1. Generate or obtain the required energy-resolved SimSET projection data.
2. Convert the SimSET projection data to the STIR-compatible format used by the analysis.
3. Update the dataset paths in `Procedure.ipynb`.
4. Run the notebook from the beginning to load the energy-resolved \(C_0-C_3\) datasets.
5. Run the relevant analysis sections for the required PDF, basis-function, spatial-region, or simulation configuration.

The Python modules should be located in the same project directory, or otherwise be available on the Python path.

---

## Notes

Monte Carlo ground-truth PDFs and direct ground-truth 2D basis functions are used only for validation and diagnostic experiments. They require knowledge of the simulated scatter history and therefore would not be directly available from measured clinical PET data.

The predicted-PDF models are used to investigate the practically applicable EBSE workflow, while the ground-truth configurations are used to identify the sources of error arising from PDF estimation, shared marginal distributions and 1D factorisation.

---

## Author

Yinyi Song

MSc Scientific and Data Intensive Computing  
University College London

## Acknowledgements

This project builds on previous work on energy-based scatter estimation in PET. Parts of the simulation and PDF-estimation workflow were adapted from earlier project code where indicated in the individual source files.

Please refer to the dissertation and source-code attribution statements for the relevant references.