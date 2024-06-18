# Code and data used in a research article on aerodynamic levitation of droplets

This repository contains the data and code used in the following research article:

[![DOI](https://doi.org/10.1002/anie.202318038)](https://doi.org/10.1002/anie.202318038)
Yankai Jia, Yaroslav I. Sobolev, Olgierd Cybulski, Tomasz Klucznik, Cristóbal Quintana, Juan Carlos Ahumada, & Bartosz A. Grzybowski,
"Aerodynamically levitated droplets as small-scale chemical reactors and liquid microprinters",
*Angewandte Chemie International Edition*, **2024**, e202318038

## Requirements

- Python 3.7
- NumPy 1.21.6
- Matplotlib 3.1.0
- Scikit-image 0.15.0
- SciPy 1.5.2

## Reproducing specific figures from the article


### Main text figures ###
To reproduce **Figure 1f**, run `plot-liquid-parameters.py`. The respective dataset is located at `misc_data/experimental_stability/2021-12-08c.xlsx`.

To reproduce **Figure 1g**, run `stability-plot.py`.The respective dataset is located at `misc_data/experimental_stability/`.

To reproduce **Figure 4c**, run `print_volumes_vs_velocity.py`. The spectral unmixing algorithms used to determine the
volumes of dyed liquid transferred into the rotating vial are implemented in `fluorimetry_processing_*.py` files.

To reproduce **Figure 6d**, run `plot_crossings_for_paper.py`. It uses datasets obtained by 
running the image processing code in `detect_tears.py` and `analyze_detected_events.py` for individual experiments
as can be seen in the `per_experiment_commands.py` script.

To reproduce **Figures 7a,b**, run `gap-vs-voltage_2.py`. The respective experimental dataset is 
located at `misc_data/wired_voltage_thresh_vs_speed.txt`. Simulation results precomputed in COMSOL are
located at `comsol_results/gap-vs-voltage/`.

To reproduce **Figure 7c**, run `plot_rheometry_2.py`. The respective experimental dataset is in `misc_data/rheometry/`.

<!-- ### Supplementary Information figures ### -->

<!-- ## Numerically solving Navier-Stokes and Laplace equation for a levitating droplet -->

