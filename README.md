# Energy-Based Scatter Estimation in PET

This repository contains the simulation-analysis code developed for an MSc research project on **energy-based scatter estimation (EBSE)** in positron emission tomography (PET).

The project investigates the estimation of scattered and unscattered coincidence events from the measured two-dimensional photon-energy distribution. Monte Carlo data generated using **SimSET** are used to evaluate different approaches for:

* estimating non-scatter and scatter energy probability density functions (PDFs);
* constructing energy basis functions;
* estimating their spatial coefficients using NEGML; and
* comparing the resulting scatter and non-scatter estimates with Monte Carlo ground truth.

The main analysis is contained in `Procedure.ipynb`, while reusable calculations are separated into Python modules according to their role in the analysis.

---

## Requirements

The analysis uses Python together with the following main packages:

* NumPy
* SciPy
* Matplotlib
* SIRF/STIR

**SimSET** is required to reproduce the Monte Carlo simulations.

The Python modules contained in this repository should either be located in the same directory as `Procedure.ipynb` or otherwise be available on the Python path.

System-specific installation and configuration instructions are not reproduced in detail here. Further information on the simulation and analysis setup is provided in the accompanying dissertation.

---

## Analysis Workflow

The main analysis follows the sequence:

```text
SimSET energy-resolved projection data
        ↓
energy-resolved C0, C1, C2, C3 components
        ↓
non-scatter and scatter energy PDFs
        ↓
2D energy basis functions
        ↓
NEGML coefficient estimation
        ↓
estimated scatter and non-scatter sinograms
        ↓
comparison with Monte Carlo ground truth
```

The four simulated coincidence components are:

* (C_0): neither detected photon scattered;
* (C_1): first photon unscattered, second photon scattered;
* (C_2): first photon scattered, second photon unscattered;
* (C_3): both detected photons scattered.

The non-scatter component is therefore represented by (C_0), while the total scatter contribution is

[
C_1 + C_2 + C_3.
]

---

## Repository Structure

### `Procedure.ipynb`

Main analysis notebook.

The notebook provides the high-level experimental workflow and includes:

* loading and preprocessing of the energy-resolved simulation data;
* global and spatially varying basis models;
* predicted and Monte Carlo ground-truth energy PDFs;
* shared and separate 1D factorised basis functions;
* direct ground-truth 2D basis functions;
* independent and axially averaged NEGML fitting;
* different spatial region sizes;
* representative-pixel diagnostics;
* sensitivity to simulated count statistics; and
* sensitivity to detector energy resolution.

Most reusable calculations are implemented in the modules below rather than repeated directly in the notebook.

### `data_loading.py`

Functions for reading and aggregating the energy-resolved SimSET/STIR projection data.

The converted projection files are separated according to the energy bins and scatter histories of the two detected photons, with filenames of the form:

```text
blue0_en[i]_pink0_en[j]
blue0_en[i]_pink1_en[j]
blue1_en[i]_pink0_en[j]
blue1_en[i]_pink1_en[j]
```

The module constructs the energy-resolved (C_0)-(C_3) arrays, their total prompt distribution (Y), and the energy-integrated ground-truth scatter and non-scatter sinograms.

### `energy_pdfs.py`

Functions for constructing one-dimensional photon-energy PDFs.

The module includes:

* derivation of Monte Carlo ground-truth non-scatter and scatter PDFs;
* prediction of the non-scatter PDF using a Gaussian detector-energy response centred at 511 keV; and
* estimation of the scatter PDF by scaling and subtracting the estimated non-scatter contribution from the total marginal energy spectrum.

The resulting PDFs are numerically normalised over the investigated energy range before basis construction.

Parts of the PDF-estimation procedure were adapted from previous EBSE code by Yifan Ding. See the [Acknowledgements](#acknowledgements) section and the attribution comments in the source file.

### `basis_functions.py`

Functions for constructing the four two-dimensional energy basis functions used by NEGML.

Three basis models are implemented.

#### Shared 1D Factorised Basis

The same non-scatter and scatter PDFs are used for both photon-energy dimensions:

[
U(E_1)U(E_2), \qquad
U(E_1)S(E_2), \qquad
S(E_1)U(E_2), \qquad
S(E_1)S(E_2).
]

#### Separate 1D Factorised Basis

Separate marginal PDFs are retained for the two photon-energy dimensions:

[
U_1(E_1)U_2(E_2), \qquad
U_1(E_1)S_2(E_2), \qquad
S_1(E_1)U_2(E_2), \qquad
S_1(E_1)S_2(E_2).
]

#### Direct Ground-Truth 2D Basis

The four Monte Carlo (C_0,C_1,C_2,C_3) joint-energy histograms are normalised directly and used as the basis functions.

This construction avoids factorising the joint energy distributions into products of one-dimensional marginal PDFs and is therefore used as a diagnostic reference.

### `negml.py`

Implementation of the NEGML coefficient-estimation procedure.

The module contains:

* construction of the 1D non-scatter/scatter system matrix;
* extraction of the two marginal prompt-energy spectra;
* variance estimation for coefficient initialisation;
* initial estimation of the four basis coefficients;
* the four-parameter NEGML update; and
* application of NEGML across the sinogram.

The four fitted coefficients correspond to the (UU), (US), (SU), and (SS) basis components.

### `spatial_regions.py`

Utilities for extracting and analysing spatially restricted energy data.

The module is used to:

* construct view--tangential spatial masks;
* extract joint energy histograms within selected regions;
* derive regional ground-truth (U_1,S_1,U_2,S_2) PDFs; and
* extract the individual (C_0)-(C_3) components for regional and representative-pixel analysis.

These functions support comparison between global and spatially varying energy basis models.

### `estimation_workflows.py`

Higher-level workflows combining PDF selection, basis construction, coefficient initialisation, and NEGML fitting.

The implemented workflows include:

* fitting axial sinogram positions independently;
* averaging energy-resolved data over the axial direction before fitting;
* estimating separate bases for rectangular spatial regions; and
* applying different PDF and basis configurations to representative pixels.

`Procedure.ipynb` calls these workflows to avoid reproducing the complete estimation procedure for each experiment.

### `Graphic_display.py`

Plotting and result-display utilities.

The module includes functions for:

* separating the four fitted NEGML components;
* combining the fitted scatter components;
* generating labels from the selected PDF and basis configuration;
* comparing estimated and ground-truth sinograms; and
* displaying individual 2D energy basis functions.

---

## Simulation Data

The projection data were generated using **SimSET** and converted to STIR-compatible energy-resolved sinograms.

The reference simulation used for the main analysis has:

* detector energy resolution: **10% FWHM at 511 keV**;
* simulated decays: **(10^9)**;
* 10 energy bins for each detected photon;
* energy range: **400--600 keV**;
* energy-bin width: **20 keV**;
* view dimension: **32**;
* tangential dimension: **55**; and
* 12 simulated detector rings.

Additional simulations with different detector energy resolutions and decay counts are used for sensitivity analyses.

The simulation datasets are stored separately from this repository because of their size.

The SimSET parameter files define the detector model, energy and projection binning, activity distribution, attenuation distribution, and photon-history output. Further details of the simulation configuration are provided in the dissertation.

---

## Spatial Basis Estimation

The analysis supports both **global** and **spatially varying** energy basis functions.

For the global model, a single set of energy PDFs and basis functions is derived from the complete (32\times55) view--tangential sinogram plane.

For the regional model, this plane is divided into rectangular regions. Separate energy information is derived within each region before NEGML is applied to the corresponding sinogram pixels.

The investigated region widths are:

```python
[1, 2, 4, 8, 16, 32]
```

The investigated region lengths are:

```python
[1, 5, 11, 55]
```

A (32\times55) region corresponds to the global spatial scale, while a (1\times1) region provides a separately derived basis for each view--tangential sinogram pixel.

---

## Running the Analysis

1. Generate or obtain the required energy-resolved SimSET projection data.
2. Convert the SimSET output to the STIR-compatible format expected by the analysis.
3. Open `Procedure.ipynb`.
4. Set `DATA_DIR` to the local directory containing the simulation datasets.
5. Run the notebook from the beginning to load and preprocess the required (C_0)-(C_3) data.
6. Run the relevant analysis sections for the required PDF, basis-function, spatial-region, or simulation configuration.

Some diagnostic NEGML calculations are computationally expensive. Precomputed results are therefore loaded by default where indicated in the notebook. These calculations can be reproduced by changing the corresponding `RUN_...` flags to `True`.

---

## Ground-Truth and Predicted Models

Monte Carlo ground-truth PDFs and direct ground-truth 2D basis functions are used only for validation and diagnostic experiments.

These quantities require knowledge of the simulated photon scatter histories and would therefore not be directly available from measured clinical PET data.

The predicted-PDF configurations represent the practically applicable EBSE workflow investigated in this project. Ground-truth configurations are used to investigate modelling errors associated with:

* non-scatter and scatter PDF estimation;
* the use of shared marginal PDFs for the two photon-energy dimensions; and
* factorisation of the joint energy distribution into products of one-dimensional PDFs.

---

## Acknowledgements

This project builds on previous work on energy-based scatter estimation in PET.

Parts of the PDF-estimation and analysis workflow were adapted from code developed by **Yifan Ding** for his UCL MSc project:

> Yifan Ding, *Evaluation of Energy-based Scatter Estimation in Positron-Emission Tomography*, MSc Precision Medicine, Department of Medicine, University College London.

Original repository:  
[https://github.com/IanDing404/Evaluation-of-Energy-based-Scatter-Estimation-in-Positron-Emission-Tomography](https://github.com/IanDing404/Evaluation-of-Energy-based-Scatter-Estimation-in-Positron-Emission-Tomography)

The adapted sections are identified in the relevant source files. Modifications by Yinyi Song are licensed under the Apache License 2.0.

---

## Author

**Yinyi Song**
MSc Scientific and Data Intensive Computing
University College London