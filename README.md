# Project Structure of the SMA_D_DREAM Framework

The project is organized into two parallel and complementary branches:  
(1) the **SMA-D-DREAM inverse modeling framework**, and  
(2) the **Transformer-based surrogate modeling module**.  

Together, these components form a surrogate-assisted Bayesian inverse modeling system designed for computationally expensive groundwater inverse model.

---

## Directory Structure

- **SMA-D-DREAM/**
  - **spotpy/**
    - Python package adapted and extended from the original **SPOTPY** library.
    - Provides the core Bayesian inference and MCMC sampling infrastructure.
    - **algorithms/**
      - `Dream_2.py`
        - Implementation of the **D-DREAM** algorithm.
        - An enhanced Differential Evolution Adaptive Metropolis (DREAM) method with convergence-aware proposal adaptation and guided differential evolution.
      - `Dream_3.py`
        - Implementation of the proposed **SMA-D-DREAM framework**.
        - Extends D-DREAM by incorporating surrogate model assistance to reduce computational cost while preserving inversion accuracy.

- **Transformer_based_surrogate_model/**
  - `GroundwaterTransformer.py`
    - Implementation of a **Transformer-based surrogate model** for groundwater simulations.
    - The model learns the nonlinear mapping between model parameters and groundwater state variables.
    - Serves as a fast approximation of the high-fidelity numerical groundwater model during the inversion process.

---

This parallel and modular design enables flexible replacement or upgrading of surrogate models and ensures the extensibility of the overall framework.
