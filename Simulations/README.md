# Simulation Data

The simulation datasets used in this project are stored separately from the GitHub repository because of their large file size.

The datasets can be accessed from the following OneDrive folder:

[Download the PET simulation data](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucaponj_ucl_ac_uk/IgAbcqoN5_ksS6Mdg6pjDn3xAdAlfceoKbClnyRC6llWKUE?e=Jrf7cX)

## Available datasets

The OneDrive folder contains the simulation datasets used for the analyses presented in this project. Each simulation is stored in a separate directory.

The original simulation directory names are retained because these names are referenced by the analysis notebooks and scripts.


## Simulation folder naming convention

The simulation folder names encode the main parameters used for each dataset. For example:

```text
sim_test_10_en_bin__1B_decays_3true_max_ring_diff_11_resolution_10
```

The components of the folder name have the following meanings:

* `10_en_bin`: the detected energy range was divided into 10 energy bins.
* `1B_decays`: (1\times10^9) source decays were simulated.
* `3true`: refers to the corresponding runtime-option setting used in the SimSET `template_phg.rec` configuration.
* `max_ring_diff_11`: the maximum allowed detector-ring difference was 11. Since the simulated scanner contains 12 detector rings, this permits coincidences between any pair of detector rings, including the two most widely separated rings.
* `resolution_10`: a detector energy resolution of 10% was used.

For example,

```text
sim_test_10_en_bin__1B_decays_3true_max_ring_diff_11_resolution_5
```

uses the same simulation settings but with a detector energy resolution of 5%.

Similarly,

```text
sim_test_10_en_bin__5M_decays_3true_max_ring_diff_11_resolution_10
```

contains (5\times10^6) simulated source decays and uses a detector energy resolution of 10%.

The original folder names are retained in the shared dataset because they are referenced directly by the analysis code and also provide a compact record of the principal simulation settings.


## Using the simulation data

After downloading the simulation data, set `DATA_DIR` in the relevant analysis notebook or script to the local directory containing the downloaded simulation folders.

For example:

```python
from pathlib import Path

# Replace this with the local path to the downloaded simulation data.
DATA_DIR = Path("/path/to/your/downloaded/PET_simulation_data")
```

Individual simulation directories are then defined relative to `DATA_DIR`. For example:

```python
reference_simulation_dir = (
    DATA_DIR
    / "sim_test_10_en_bin__1B_decays_3true_max_ring_diff_11_resolution_10"
)
```

Only `DATA_DIR` needs to be changed to match the location of the downloaded data on the user's system.

Please retain the original simulation folder names after downloading the datasets.

## Simulation generation

The SimSET parameter files and scripts used to generate and process the simulation data are included in this repository.

These include the activity and attenuation phantom-generation files, SimSET configuration templates, simulation scripts, and conversion scripts required for processing the simulated projection data.

Generated activity and attenuation volume files and other large simulation outputs are not stored directly in the GitHub repository.