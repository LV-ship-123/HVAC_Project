# HVAC Fitting & Optimization Tool

**Author**: june  
**Date**: September 2026  
**Repository**: [LV-ship-123/HVAC_Project](https://github.com/LV-ship-123/HVAC_Project)

---

## 1. Project Overview

This project started as a reproduction of **Figure 8a** from the paper:  
*"基于HOA的建筑集中空调系统冷却塔出水温度优化控制研究"* (Research on Optimal Control of Cooling Tower Outlet Temperature in Central Air-Conditioning Systems Based on HOA).

It has since evolved into a **complete physical modeling and optimization pipeline**, including:
- Polynomial fitting and gradient-based optimization
- Genetic algorithm (GA) verification
- Physical model construction (energy balance + Merkel ODE)
- Robustness analysis under parameter perturbation
- Engineering constraint integration (penalty functions)

---

## 2. Methodology Evolution (01 → 15)

| Phase | Code | Core Idea | Key Result |
| :--- | :--- | :--- | :--- |
| **1. Baseline Reproduction** | `01_plot_fig8a.py` | Fitted 4 data points with a quadratic polynomial | Reproduced Fig 8a exactly |
| **2. Tool Encapsulation** | `02_my_toolkit.py` | Wrapped fitting logic into a reusable function | Portable fitting tool |
| **3. Optimization Understanding** | `04_search_optimal.py` | Brute-force search across [30, 33] with 0.01°C step | Verified that 30°C is the boundary optimum |
| **4. Gradient-Based Solver** | `05_optimize_scipy.py` | Used L-BFGS-B for fast convergence | Same result, 10× faster |
| **5. Genetic Algorithm** | `07_genetic_algorithm.py` | Applied GA to verify optimizer independence | GA found 30°C with only 34 evaluations |
| **6. First Physical Model** | `08_merkel_model.py` | U-shaped curve: compressor vs pump trade-off | Internal optimum at **31.40°C** |
| **7. Robustness Test** | `10_robustness_analysis.py` | 50 random perturbations (±20% on parameters) | Mean 31.35°C, std 0.17°C → robust |
| **8. Energy Balance Model** | `13_merkel_true_physical.py` | Actual heat transfer limited by the smaller of water-side and air-side capacity | Air flow finally participates in cooling physics |
| **9. Engineering Penalty** | `14_merkel_with_penalty.py` | Added penalty for m_water < 4.0 kg/s | Optimum moved to **3.97 kg/s** (feasible region) |
| **10. Merkel ODE Coupled** | `15_merkel_ode_coupled.py` | Coupled ODE: water temp + air enthalpy | True physics-driven optimum: **2.96 kg/s, 34.02°C** |

---

## 3. Key Results

- **Polynomial Fitting**: `y = -0.125x² + 9.025x - 50.225`
- **Mathematical Optimum**: `30.00 °C` (boundary optimum)
- **Physical Optimum (Energy Balance)**: `31.40 °C` (internal trade-off)
- **Physics-Driven Optimum (Merkel ODE)**: `2.96 kg/s` water flow, `34.02 °C` outlet temp

> The final model is based on **Merkel enthalpy-difference theory**, solving coupled ODEs for water temperature and air enthalpy along the packing height. This is the same framework used in peer-reviewed journal papers on cooling tower optimization.

---

## 4. File Structure

```
HVAC_Project/
├── 01_plot_fig8a.py            # Step 1: Basic reproduction
├── 01_Fig8a_base.png           # Output of Step 1
├── 02_my_toolkit.py            # Step 2: Reusable fitting function
├── 02_Fitted_verify.png        # Output of Step 2 (test the toolkit)
├── 03_test_my_tool.py          # Step 3: Test the toolkit
├── 04_search_optimal.py        # Step 4: Brute-force validation
├── 05_optimize_scipy.py        # Step 5: L-BFGS-B optimizer
├── 06_plot_with_optimum.py     # Step 6: Visualization with optimum point
├── 06_Fig8a_optimal.png        # Final output (with green star)
├── 07_genetic_algorithm.py     # Step 7: Genetic algorithm verification
├── 08_merkel_model.py          # Step 8: First physical trade-off model
├── 09_compare_methods.py       # Step 9: Method comparison (L-BFGS-B vs GA)
├── 10_robustness_analysis.py   # Step 10: Robustness test (50 random perturbations)
├── 11_merkel_ode.py            # Step 11: ODE attempt (air flow bug)
├── 12_merkel_physical_fixed.py # Step 12: ε-NTU fix (still flawed)
├── 13_merkel_true_physical.py  # Step 13: Energy balance (correct physics)
├── 14_merkel_with_penalty.py   # Step 14: Feasibility penalty
├── 15_merkel_ode_coupled.py    # Step 15: Merkel ODE coupled (final)
├── 07_Physical_Optimum.png     # Output of Step 7/8 (physical model)
├── 08_Method_Comparison.png    # Output of Step 9 (method comparison)
├── 09_Robustness_Distribution.png # Output of Step 10 (robustness)
├── 10_Merkel_ODE_Optimization.png # Output of Step 11 (ODE attempt)
├── 11_Merkel_Fixed_Optimization.png # Output of Step 12 (ε-NTU fix)
├── 12_True_Physical_Optimization.png # Output of Step 13 (true physical)
├── 13_Merkel_With_Penalty.png  # Output of Step 14 (penalty)
├── 14_Merkel_ODE_Coupled.png   # Output of Step 15 (final ODE)
└── README.md                   # This file
```

---

## 5. Dependencies

- Python 3.x
- numpy
- matplotlib
- scipy

Install via:

```bash
pip install numpy matplotlib scipy
6. How to Run
git clone https://github.com/LV-ship-123/HVAC_Project.git
cd HVAC_Project
python 15_merkel_ode_coupled.py
7. Future Work
Extend to time-varying wet-bulb temperature (dynamic optimization)

Connect with EnergyPlus for real building load data

Replace penalty with multi-objective optimization (Pareto front)

8. License
Academic and research purposes only.
