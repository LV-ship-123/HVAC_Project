# HVAC Fitting & Optimization Tool

**Author**: june  
**Date**: September 2026  
**Repository**: [LV-ship-123/HVAC_Project](https://github.com/LV-ship-123/HVAC_Project)

---

## 1. Background

This project reproduces **Figure 8a** from the paper:  
*"基于HOA的建筑集中空调系统冷却塔出水温度优化控制研究"* (Research on Optimal Control of Cooling Tower Outlet Temperature in Central Air-Conditioning Systems Based on HOA).

The original figure shows the relationship between cooling tower outlet temperature setpoint and total power consumption.

---

## 2. Methodology

- **Data Extraction**: 4 key data points were extracted:  
  `(30°C, 108 kW)`, `(31°C, 109.5 kW)`, `(32°C, 110.5 kW)`, `(33°C, 111.5 kW)`.
- **Polynomial Fitting**: Quadratic polynomial (`degree = 2`) via `numpy.polyfit`.
- **Optimization**: 
  - *Brute-force*: Exhaustively scanned [30, 33] with 0.01°C step.
  - *SciPy Optimizer*: L-BFGS-B method for verification.
- **Visualization**: Optimal point marked with a green star.

---

## 3. Results

- Fitted polynomial: `y = -0.125x² + 9.025x - 50.225`
- **Optimal Temperature**: `30.00 °C`
- **Minimum Power**: `108.03 kW`

> The optimal setpoint is at the lower boundary (30°C), confirming that power consumption increases with temperature in this range.

---

## 4. File Structure (Ordered by Workflow)

```
HVAC_Project/
├── 01_plot_fig8a.py          # Step 1: Basic reproduction
├── 02_my_toolkit.py          # Step 2: Reusable fitting function
├── 03_test_my_tool.py        # Step 3: Test the toolkit
├── 04_search_optimal.py      # Step 4: Brute-force validation
├── 05_optimize_scipy.py      # Step 5: Professional optimizer
├── 06_plot_with_optimum.py   # Step 6: Final visualization
├── 01_Fig8a_base.png         # Output of Step 1
├── 06_Fig8a_optimal.png      # Final output (with green star)
└── README.md                 # This file
```

## 5. Dependencies

- Python 3.x
- numpy, matplotlib, scipy

Install:
```bash
pip install numpy matplotlib scipy
```

## 6. How to Run

```bash
git clone https://github.com/LV-ship-123/HVAC_Project.git
cd HVAC_Project
python 06_plot_with_optimum.py
```

## 7. Future Work
- Replace polynomial fitting with a physical Merkel equation model.
- Implement Genetic Algorithm (GA) for global optimization.

## 8. License
Academic and research purposes only.