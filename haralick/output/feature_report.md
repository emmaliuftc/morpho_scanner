# NeRF Morphometric Feature Report

**Generated:** 2026-09-23 23:58:28

**Method:** Ported from `polymorpho-lee-lab/features.py` + `hierarchical.py`

## Feature Definitions

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 0 | Volume | `features.py:81` | Voxel count after hole-filling |
| 1 | Surface Area | `features.py:101-103` | Marching cubes mesh area |
| 2 | Sphericity | `features.py:108` | π^(1/3) × (6V)^(2/3) / SA |
| 3 | Lobe Count | `hierarchical.py:118-253` | Recursive erosion → dendrogram |
| 4–16 | Haralick [1–13] | `features.py:91` | 13 GLCM texture features (mean over 13 3D directions) |

### Haralick Feature Names

| Index | Name | Abbreviation |
|-------|------|--------------|
| 4 | ASM | ASM |
| 5 | Contrast | Contrast |
| 6 | Correlation | Correlation |
| 7 | Sum_of_Squares_Variance | Sum of Sq: Var |
| 8 | IDM | IDM |
| 9 | Sum_Average | Sum Avg |
| 10 | Sum_Variance | Sum Var |
| 11 | Sum_Entropy | Sum Entropy |
| 12 | Entropy | Entropy |
| 13 | Diff_Variance | Diff Var |
| 14 | Diff_Entropy | Diff Entropy |
| 15 | IMC_I | IMC-I |
| 16 | IMC_II | IMC-II |

## Feature Matrix

| Specimen | Volume | Surface Area | Sphericity | Lobes | ASM | Contrast | Corr. | SoS:V | IDM | Sum Avg | Sum Var | Sum Ent | Ent | Diff Var | Diff Ent | IMC-I | IMC-II |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Hemisphere (20k)** | 738,417 | 62,087.1 | 0.6363 | 4 | 0.4728 | 0.0281 | 0.9438 | 0.2500 | 0.9859 | 1.0051 | 0.9719 | 1.1526 | 1.1807 | 0.2228 | 0.1807 | -0.8193 | 0.8967 |
| **Volume One** | 473,791 | 45,355.5 | 0.6480 | 2 | 0.5944 | 0.0163 | 0.9582 | 0.1948 | 0.9918 | 0.5300 | 0.7628 | 0.9345 | 0.9508 | 0.2340 | 0.1196 | -0.8603 | 0.8728 |
| **Volume Three** | 356,795 | 162,212.1 | 0.1500 | 1 | 0.6078 | 0.0619 | 0.8150 | 0.1673 | 0.9690 | 0.4247 | 0.6071 | 0.9929 | 1.0549 | 0.1924 | 0.3285 | -0.5861 | 0.7587 |
| **Four Flat Full** | 279,543 | 252,621.8 | 0.0818 | 1 | 0.5986 | 0.1147 | 0.6228 | 0.1516 | 0.9426 | 0.3725 | 0.4916 | 1.0130 | 1.1277 | 0.1517 | 0.4851 | -0.3753 | 0.5961 |
| **Lesion 3 (20k)** | 54,950 | 34,139.0 | 0.2047 | 3 | 0.6741 | 0.0646 | 0.7571 | 0.1329 | 0.9677 | 0.3155 | 0.4669 | 0.8740 | 0.9386 | 0.1898 | 0.3423 | -0.5078 | 0.6837 |
| **Lesion 3 (200k)** | 48,022 | 26,629.8 | 0.2399 | 2 | 0.7149 | 0.0513 | 0.7834 | 0.1183 | 0.9744 | 0.2742 | 0.4219 | 0.7846 | 0.8358 | 0.2015 | 0.2894 | -0.5505 | 0.6832 |
| **Lesion 4** | 92,553 | 68,361.3 | 0.1447 | 3 | 0.9872 | 0.0026 | 0.7467 | 0.0051 | 0.9987 | 0.0103 | 0.0179 | 0.0622 | 0.0648 | 0.2474 | 0.0258 | -0.6061 | 0.2335 |

## Analysis

### Volume Comparison

| Specimen | Volume (voxels) | Relative to Hemisphere |
|----------|----------------|------------------------|
| Hemisphere (20k) | 738,417 | 1.00× |
| Volume One | 473,791 | 0.64× |
| Volume Three | 356,795 | 0.48× |
| Four Flat Full | 279,543 | 0.38× |
| Lesion 3 (20k) | 54,950 | 0.07× |
| Lesion 3 (200k) | 48,022 | 0.07× |
| Lesion 4 | 92,553 | 0.13× |

### Lesion 3 vs Lesion 4 Comparison

| Feature | Lesion 3 (20k) | Lesion 3 (200k) | Lesion 4 | L4/L3_200k Ratio |
|---------|---------------|-----------------|----------|------------------|
| Volume_voxels | 54,950 | 48,022 | 92,553 | 1.93× |
| Surface_Area | 34138.9570 | 26629.8145 | 68361.3359 | 2.57× |
| Sphericity | 0.2047 | 0.2399 | 0.1447 | 0.60× |
| Lobe_Count | 3 | 2 | 3 | 1.50× |
| ASM | 0.6741 | 0.7149 | 0.9872 | 1.38× |
| Contrast | 0.0646 | 0.0513 | 0.0026 | 0.05× |
| Correlation | 0.7571 | 0.7834 | 0.7467 | 0.95× |
| Sum_of_Squares_Variance | 0.1329 | 0.1183 | 0.0051 | 0.04× |
| IDM | 0.9677 | 0.9744 | 0.9987 | 1.02× |
| Sum_Average | 0.3155 | 0.2742 | 0.0103 | 0.04× |
| Sum_Variance | 0.4669 | 0.4219 | 0.0179 | 0.04× |
| Sum_Entropy | 0.8740 | 0.7846 | 0.0622 | 0.08× |
| Entropy | 0.9386 | 0.8358 | 0.0648 | 0.08× |
| Diff_Variance | 0.1898 | 0.2015 | 0.2474 | 1.23× |
| Diff_Entropy | 0.3423 | 0.2894 | 0.0258 | 0.09× |
| IMC_I | -0.5078 | -0.5505 | -0.6061 | 1.10× |
| IMC_II | 0.6837 | 0.6832 | 0.2335 | 0.34× |

### Lesion 3: 20k vs 200k Training Steps

| Feature | 20k Steps | 200k Steps | Change |
|---------|-----------|------------|--------|
| Volume_voxels | 54,950 | 48,022 | -12.6% |
| Surface_Area | 34138.9570 | 26629.8145 | -22.0% |
| Sphericity | 0.2047 | 0.2399 | +17.2% |
| Lobe_Count | 3 | 2 | -33.3% |
| ASM | 0.6741 | 0.7149 | +6.1% |
| Contrast | 0.0646 | 0.0513 | -20.6% |
| Correlation | 0.7571 | 0.7834 | +3.5% |
| Sum_of_Squares_Variance | 0.1329 | 0.1183 | -11.0% |
| IDM | 0.9677 | 0.9744 | +0.7% |
| Sum_Average | 0.3155 | 0.2742 | -13.1% |
| Sum_Variance | 0.4669 | 0.4219 | -9.6% |
| Sum_Entropy | 0.8740 | 0.7846 | -10.2% |
| Entropy | 0.9386 | 0.8358 | -10.9% |
| Diff_Variance | 0.1898 | 0.2015 | +6.2% |
| Diff_Entropy | 0.3423 | 0.2894 | -15.5% |
| IMC_I | -0.5078 | -0.5505 | -8.4% |
| IMC_II | 0.6837 | 0.6832 | -0.1% |

### Sphericity Ranking (1.0 = perfect sphere)

| Rank | Specimen | Sphericity |
|------|----------|------------|
| 1 | Volume One | 0.6480 |
| 2 | Hemisphere (20k) | 0.6363 |
| 3 | Lesion 3 (200k) | 0.2399 |
| 4 | Lesion 3 (20k) | 0.2047 |
| 5 | Volume Three | 0.1500 |
| 6 | Lesion 4 | 0.1447 |
| 7 | Four Flat Full | 0.0818 |

### Observations

1. **Hemisphere** should have the highest sphericity (~1.0 for a perfect half-sphere).
2. **Lesion 3 vs Lesion 4**: The user notes they should have similar or 2× volume.
3. **20k vs 200k training**: More training iterations should yield tighter, more accurate geometry.
4. **Haralick on binary**: Since our volumes are binary (0/1), Haralick features reflect the *spatial distribution* of the occupied voxels rather than intensity texture.
5. **Lobe Count**: Uses the exact same recursive erosion → hierarchical dendrogram algorithm from `hierarchical.py`.
