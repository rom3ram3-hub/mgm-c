# MGM-C: Calibrated Self-Supervised Masked Gate Modeling for Trojan Detection in Untrusted Quantum Compilation

Code and results for the paper

> S. J. Basha, T. Ammannamma, D. Veeraiah, M. H. Krishna, S. N. Mohanty, and P. Suhas, "MGM-C: Calibrated self-supervised masked gate modeling for trojan detection in untrusted quantum compilation," in *Proc. IEEE Int. Conf. Consumer Electronics (ICCE)*, Las Vegas, NV, USA, 2027.

## Overview

Users of cloud quantum services send logical circuits to third-party compilers and receive hardware-native circuits in return. A malicious compiler can insert a few gates into the returned circuit. Existing learning-based detectors are supervised, so they recognize only the trojan types they were trained on.

**MGM-C** learns what honest compiler output looks like from clean circuits alone, and flags gates that do not fit:

1. Each compiled circuit is converted to a **port graph**, in which every (gate, qubit) incidence is a node. Masking a gate hides its arity, so a masked `cx` looks the same as two masked single-qubit gates.
2. A four-layer **GINE** network is trained to predict masked gates (masked gate modeling, 15% masking).
3. At inference, every gate is scored by its **surprisal** while hidden. The surprisal is converted into a **conformal-style p-value per gate type**, and the circuit score applies a Bonferroni correction for circuit size.
4. The per-gate p-values also rank gates for **localization**.

No trojan examples are used at any stage of training or calibration.

## Main results

| Setting | Result |
|---|---|
| Unseen trojan types, 10 family splits | MGM-C mean AUC **0.613 ± 0.058** vs. B4 0.538 and GIN-L2D 0.510; paired gain over B4 **0.075** (95% CI 0.049 to 0.093), 9 of 10 splits |
| Seen trojan types, family-held-out 5-fold CV | GIN-L2D 0.689, B4 0.669, MGM-C 0.621, label-free fusion 0.693 |
| Evaluation realism (2 × 2 study) | Random splits over a fixed compiler inflate AUC by up to 0.35 |
| Localization | MGM-C ranks a true trojan gate first in 26.2% of trojaned circuits (random: 2.3%) |
| Cross-compiler (tket) | Recalibrating MGM-C references on tket output, without retraining, reduces the false-positive rate from 0.24–0.34 to 0.104 |
| Adaptive adversary | Every reference-free detector, including an ensemble, reaches chance level within 8 to 32 placement attempts |
| CPU cost | Median 10 ms (B4), 16 ms (GIN-L2D) and 30 ms (MGM-C) per circuit on a two-thread CPU |

Note that the GPU training of the neural models is not bit-for-bit deterministic, so a fresh run can differ from these values in the second or third decimal place. The tree-based baselines (B1–B4) are fully deterministic.

## Repository contents

```
├── QTrojan_GNN_Detection_v6.ipynb   # complete pipeline: data, models, all experiments
├── make_figures.py                  # redraws paper Figs. 2–5 from the CSV results
├── results/                         # CSV tables and diagnostic plots from the run reported in the paper
├── figures/                         # paper figures (Fig. 1 pipeline diagram and Figs. 2–5)
└── README.md
```

## Requirements

The notebook is designed for **Google Colab with a T4 GPU**. It installs its own dependencies in the first cell:

| Package | Version |
|---|---|
| qiskit | 2.5.2 |
| qiskit-aer | 0.17.2 |
| mqt.bench | 2.3.0 |
| torch, torch_geometric | Colab default |
| pytket | latest (used only for the cross-compiler test) |
| scikit-learn, pandas, numpy, matplotlib | Colab default |

A local run needs Python 3.10 or later, a CUDA GPU (recommended) and the packages above.

## Data

No dataset needs to be downloaded by hand. The notebook builds the dataset from two public benchmark suites:

- **MQT Bench** 2.3.0, target-independent level, 3 to 10 qubits: <https://github.com/cda-tum/mqt-bench>
- **QASMBench**, small and medium circuits, cloned automatically: <https://github.com/pnnl/QASMBench>

Only static unitary circuits are retained (no mid-circuit measurement, reset or classical control). Compilations with more than 1,500 gates are discarded. The final dataset has **211 programs in 34 algorithm families**, and **2,086 compiled circuits**, of which 1,043 are trojaned. All compilation and insertion steps use fixed seeds.

## How to run

1. Open `QTrojan_GNN_Detection_v6.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. *(Optional, recommended on free Colab.)* To keep results if the session disconnects, add this cell after the first one:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   !mkdir -p /content/drive/MyDrive/qtrojan_v6_results
   !rm -rf results && ln -s /content/drive/MyDrive/qtrojan_v6_results results
   ```
   Otherwise all results are written to `./results`.
4. Select **Runtime → Run all**. A full run takes about 2 to 2.5 hours on a T4.
5. The last cell writes `qtrojan_results_v6.zip` with all tables and plots.

To redraw the paper figures from the results:

```bash
python make_figures.py --results results --out figures
```

## Notebook sections and paper mapping

| Notebook section | Experiment | Paper | Main output files |
|---|---|---|---|
| 1–4 | Benchmarks, honest compiler, trojan library (T1–T5), dataset and trojan impact | Sec. III-A, Table I | `table1_dataset.csv`, `table2_trojan_impact.csv`, `dataset_metadata.csv` |
| 5–8 | Gate graph, baselines B1–B4, GIN-L2D, family-held-out 5-fold CV (E1) | Table II (seen types) | `table3_main_detection.csv`, `e1_predictions.csv` |
| 10 | Localization (E4) | Table V | `table6_localisation.csv` |
| 13b | Evaluation realism, 2 × 2 study (E6) | Fig. 2, Sec. IV-A | `table9_realism_2x2_auc.csv` |
| 13c | MGM and MGM-C (E7): 5-fold CV, gate-level evidence, MGM-C localization | Tables II and V, Fig. 4, Sec. IV-C/D | `table10_mgm_auc_by_type.csv`, `table11_mgm_recall_localisation.csv`, `table15_gate_surprisal.csv`, `e7_mgm_predictions.csv` |
| 13d | Adaptive attacker and ensemble (E8) | Fig. 5, Sec. IV-F | `table16_attack_budget_curve.csv`, `table13_attack_transfer_auc.csv` |
| 15 R1 | Unseen trojan types over 10 family splits, bootstrap CIs, unsupervised baselines U1–U3, false-positive rates | Tables II–IV, Fig. 3 | `tableR1_unseen_auc_10splits.csv`, `tableR1_paired_difference_ci.csv`, `tableR1_fpr_unseen_families.csv`, `r1_unseen_auc_per_split.csv` |
| 15 R2 | Ablations: port vs. gate graph, mask rate, depth, calibration variants | Table IV | `tableR2_ablations.csv` |
| 15 R3 | Angle-matched trojans and insertion before optimization | Sec. IV-B | `tableR3_hard_trojans_auc.csv` |
| 15 R4 | Cross-compiler test with tket and label-free recalibration (programs whose tket output still contains SWAP gates are skipped; 196 programs remain) | Sec. V | `tableR4_cross_compiler_tket.csv` |
| 15 R5 | CPU inference time | Sec. IV-F | `tableR5_cpu_latency.csv` |
| 15 R6 | Impact from several inputs; detection by impact, stratified by trojan type | Table VI, Sec. IV-E | `tableR6_impact_by_type_auc.csv`, `tableR6_multi_input_dormancy.csv` |

## Threat model and scope

- The defender sees only the returned circuit in the native basis {rz, sx, x, cx}. The source circuit, a reference compilation and full simulation are not available to it.
- Trojans are inserted into the compiled circuit after the optimization passes.
- Circuits have at most 11 qubits, and hardware noise is not modeled.

## Citation

```bibtex
@inproceedings{basha2027mgmc,
  author    = {Basha, Shaik Johny and Ammannamma, T. and Veeraiah, Duggineni and Krishna, Malempati Hari and Mohanty, Sachi Nandan and Suhas, Penumudi},
  title     = {{MGM-C}: Calibrated Self-Supervised Masked Gate Modeling for Trojan Detection in Untrusted Quantum Compilation},
  booktitle = {Proc. IEEE International Conference on Consumer Electronics (ICCE)},
  address   = {Las Vegas, NV, USA},
  year      = {2027}
}
```

## License

Released under the MIT License (see `LICENSE`). The benchmark circuits remain under the licenses of MQT Bench and QASMBench.

## Contact

Shaik Johny Basha, Department of CSE, SRM University-AP, Amaravati, India: johnybasha.s@srmap.edu.in
