# tools/ index

Run everything from the repository root. The step scripts in `reproduce/` call these tools; see `REPRODUCE.md`.
"Upstream" = unchanged (or minimally changed) from `ostadabbas/Infant-Pose-Estimation` @ `a76b4f4`.

## Training
| Script | Origin | Used for |
|---|---|---|
| `train_adaptive_model_hrnet.py` | upstream, plus optional W&B and a per-epoch `val AP` line | every HRNet run (D1, F*, G*, H*, L*, S*, SY*) |
| `train_adaptive_model_hrnet_x.py` | copy of the above, plus `LOSS.USE_OHKM` | B3 (OHKM), B4 |
| `train_adaptive_model_mobile_wb.py` | upstream mobile script, plus W&B | M1, M2 |
| `train_adaptive_model.py`, `train_adaptive_model_mobile.py`, `train.py` | upstream, unused | — |

## Evaluation
| Script | Origin | What it does |
|---|---|---|
| `test_adaptive_model.py` | upstream | official eval: GT boxes, flip test, COCO OKS AP; writes `results/keypoints_validate_infant_results_0.json` |
| `eval_ensemble_tta.py` | ours | heatmap-level model ensemble + scale/rotation TTA on the same decode/eval path (A0–A3) |
| `eval_subsets.py` | ours | rescores saved predictions on Test100 / Test99 / Test95 / Test500f / Test499f (`--pred`, `--pred500` pick the prediction roots) |
| `report_stats_bootstrap.py` | ours | full AP/AP50/AP75/AR + paired image bootstrap vs the released model (REPORT §5.1) |
| `eval_test96.py` | ours | superseded by `eval_subsets.py` |
| `test.py` | upstream, unused | — |

## Data preparation and audit
| Script | What it builds / checks |
|---|---|
| `make_syrip_fixed.sh` | `data/syrip_fixed/`: correct synthetic train labels + correct `test11.jpg` |
| `make_syrip500_fixed.py` | `data/syrip500_fixed/`: Test500 with area = bbox area |
| `../slurm/make_real_only_data.sh`, `../slurm/make_plus_syn_data.sh` | `data/syrip_fixed_200R/` (G2), `data/syrip_fixed_plus522/` (SY1/SY2) |
| `check_annotations.py` | annotation audit (structure, bbox/area, duplicates, label agreement between copies) |

## Figures and analysis
| Script | Output |
|---|---|
| `plot_curves.py` | `figures/<run>_curves.png` from `log/<run>/` TensorBoard events |
| `plot_compare.py` | `figures/compare_4_5.png` (E1b vs D1 vs F3 vs F4) |
| `plot_report_ablation.py` | `figures/report/ablation_overview.png` (REPORT Fig. 1) |
| `visualize_preds.py` | `figures/viz_final/`: GT vs prediction grids, per-joint OKS |

## Demos and applications
| Script | What it does |
|---|---|
| `fidip_infer.py` | shared single-image inference, with an optional Faster R-CNN person detector |
| `demo.py` | CLI demo (guide Part 6.1) |
| `streamlit_demo.py` | web demo (Part 6.2): `streamlit run tools/streamlit_demo.py` |
| `video_infer.py` | video file / folder / webcam with box tracking (Part 7.3) |
| `wandb_smoke.py` | offline check of the W&B code path |

Synthetic data generation with SMIL (Part 7.1) lives in `../syn_generation/tools/`; see `docs/part7_synthetic_and_apps.md`.
