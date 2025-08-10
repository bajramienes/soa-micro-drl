# DRL-Based Orchestration Framework for SOA and Microservices

This repository contains the implementation, results, and visualizations for the research on **Deep Reinforcement Learning (DRL)-based orchestration for sustainable cloud computing using Service-Oriented Architecture (SOA) and Microservices**.

All experiments were performed in real-time using Docker-based orchestration, without simulations. The dataset, code, and visualizations are made publicly available to ensure academic reproducibility and future comparative studies.

---

## 📂 Repository Structure

```
soa-microservices/
│
├── __pycache__/                   # Compiled Python cache files
│   └── drl_env.cpython-313.pyc
│
├── Images/                        # All result charts in vector (PDF) format
│   ├── action_freq_comparison.pdf
│   ├── action_heatmap_binned.pdf
│   ├── action_rate_binned.pdf
│   ├── average_metrics_bar.pdf
│   ├── avg_energy_grouped.pdf
│   ├── avg_latency_grouped.pdf
│   ├── avg_reward_grouped.pdf
│   ├── avg_throughput_grouped.pdf
│   ├── box_plot_metrics.pdf
│   ├── correlation_heatmap_microservices.pdf
│   ├── correlation_heatmap_soa.pdf
│   ├── energy_line_plot.pdf
│   ├── energy_violin_plot.pdf
│   ├── energy_vs_throughput_microservices_hexbin.pdf
│   ├── energy_vs_throughput_soa_hexbin.pdf
│   ├── latency_histogram.pdf
│   ├── latency_line_plot.pdf
│   ├── latency_violin_plot.pdf
│   ├── latency_vs_throughput_scatter.pdf
│   ├── ppo_action_frequency_stacked.pdf
│   ├── ppo_cumulative_reward.pdf
│   ├── ppo_reward_distribution_box.pdf
│   ├── radar_early.pdf
│   ├── radar_final.pdf
│   ├── radar_mid.pdf
│   ├── throughput_histogram.pdf
│   └── throughput_line_plot.pdf
│
├── results/                       # Experiment results (JSON/ZIP logs)
│   ├── baseline_early_random.json
│   ├── baseline_early_static_micro.json
│   ├── baseline_early_static_soa.json
│   ├── baseline_final_random.json
│   ├── baseline_final_static_micro.json
│   ├── baseline_final_static_soa.json
│   ├── baseline_mid_random.json
│   ├── baseline_mid_static_micro.json
│   ├── baseline_mid_static_soa.json
│   ├── baseline_summary.json
│   ├── early_phase_test_log.json
│   ├── mid_phase_test_log.json
│   ├── final_phase_test_log.json
│   ├── ppo_early.zip
│   ├── ppo_mid.zip
│   ├── ppo_final.zip
│   └── step_logs.jsonl
│
├── tensorboard_logs/               # TensorBoard training logs
│   └── PPO_0/
│       ├── events.out.tfevents.*
│
├── baseline.py                     # Baseline orchestration implementation
├── drl_env.py                      # DRL environment definition
├── gen_chart.py                    # Script to generate result charts
├── gen_chart_log.py                 # Script to generate log-based charts
├── logs - baseline/                # Baseline logs
├── logs-drl/                        # DRL orchestration logs
├── test_docker_connection.py        # Docker connection test script
├── traing_agent.py                  # DRL training script
└── README.md
```

---

## 📊 Reproducibility

All scripts and vector charts are available at:

🔗 **[GitHub Repository] https://github.com/bajramienes/soa-micro-drl**

Researchers can directly run the provided scripts to reproduce all figures and metrics.

---


## 📬 Contact

**Enes Bajrami**  
PhD Candidate in Software Engineering and Artificial Intelligence  
Faculty of Computer Science and Engineering, Ss. Cyril and Methodius University
Skopje, North Macedonia
Email: enes.bajrami@students.finki.ukim.mk
```

