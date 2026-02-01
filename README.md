# SMA-D-DREAM Project Structure Description

The **SMA-D-DREAM** directory contains the core implementation of the proposed inverse modeling framework.  
This project is built upon the **SPOTPY** (Statistical Parameter Optimization Tool for Python) package and extends its original DREAM algorithm.

## Directory Structure

- **SMA-D-DREAM/**
  - **spotpy/**
    - Python package adapted and extended from the original SPOTPY library.
    - Contains all necessary components for Bayesian inference and MCMC-based parameter estimation.
    - **algorithms/**
      - `Dream_2.py`
        - Implementation of the **D-DREAM** algorithm.
        - This version introduces adaptive mechanisms based on convergence diagnostics and guided differential evolution.
      - `Dream_3.py`
        - Implementation of the proposed **SMA-D-DREAM framework**.
        - Integrates surrogate model assistance with the D-DREAM algorithm to significantly improve computational efficiency while maintaining inversion accuracy.

## Algorithm Roles

- **DREAM_2 (D-DREAM)**  
  Implements an enhanced Differential Evolution Adaptive Metropolis algorithm 

- **DREAM_3 (SMA-D-DREAM Framework)**  
  Extends D-DREAM by incorporating surrogate models to accelerate likelihood evaluations, forming a **surrogate model–assisted inverse modeling framework** suitable for computationally expensive numerical models.


