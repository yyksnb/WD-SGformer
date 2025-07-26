
# WD-SGformer: Weather Differentiated Spatial Graph Former for Wind Power Forecasting

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the **WD-SGformer** model, a novel deep learning architecture for multi-step wind power forecasting, as presented in our paper:

> **[ WD-SGformer: high-precision wind power forecasting via dual-attention dynamic spatio-temporal learning]**  
> *[Yakai Yang], et al.*  
> *[Under Review]* (or *Under Review*)  
> **[Link to your paper - e.g., on ArXiv, once available]**

---

## Core Idea

WD-SGformer effectively models complex spatiotemporal dependencies in wind power systems by decoupling them into two specialized, innovative encoders:

1.  **Temporal Encoder**: This encoder introduces a novel **Weather Differentiated Attention** mechanism. Instead of treating all weather information uniformly, it learns a customized weather context for each individual wind turbine. This allows the model to capture fine-grained temporal dynamics by understanding how local weather uniquely impacts each turbine.

2.  **Spatial Encoder**: This encoder features an innovative **Spatial Graph Attention** mechanism. It enhances standard self-attention by incorporating two powerful biases into the attention score calculation:
    *   **Static Geographic Bias**: Encodes the fixed geographical distances between nodes, providing the model with inherent spatial awareness.
    *   **Dynamic Graph Bias**: Captures the evolving relationships based on the temporal patterns of node features (e.g., wind speed, power output), allowing the graph structure to adapt dynamically over time.

This dual-encoder architecture enables WD-SGformer to achieve state-of-the-art performance in wind power forecasting tasks.

*(Optional: You can add a link or an image of your model architecture here for better visualization, for example:)*  
*![Model Architecture](path/to/your/architecture_diagram.png)*<img width="1062" height="531" alt="image" src="https://github.com/user-attachments/assets/fcf58ec4-2c69-4bd8-830e-57e9863d0cb8" />


---

## Project Structure

The repository is organized to clearly separate the core model from demonstration code:

```
/WDSGformer/
├── demo.py                 # Main script to demonstrate model usage
├── model/
│   ├── __init__.py         # Makes model components easily importable
│   ├── wd_sgformer.py      # The main WDSGformer model architecture
│   ├── temporal_encoder.py # Modules for the Temporal Encoder
│   └── spatial_encoder.py  # Modules for the Spatial Encoder
├── sample_data/
│   ├── turbine_info.csv    # Anonymized static turbine info
│   ├── scada_data.csv      # Anonymized turbine time-series data
│   └── weather_data.csv    # Anonymized weather time-series data
├── requirements.txt        # Required Python packages
└── README.md               # This file
```

---

## Quick Start

This repository includes a small, anonymized sample of the dataset in the `/sample_data` directory, allowing you to run a demonstration out of the box.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/WDSGformer.git
cd WDSGformer
```

### 2. Install dependencies

It is recommended to use a virtual environment (e.g., `conda` or `venv`).

```bash
pip install -r requirements.txt
```

---

## 📖 Note on Reproducibility

This repository provides the core implementation of the WD-SGformer model architecture for verification and academic use. It includes all necessary model components and a demonstration script (`demo.py`) to illustrate a forward pass.

Due to intellectual property considerations and this work being part of a larger, ongoing project, the complete training pipeline, full data preprocessing scripts, and specific hyperparameter configurations are not included in this release. We believe the provided model code is sufficient to validate the novelty and correctness of our proposed architecture.

For any questions regarding the implementation, please feel free to open an issue or contact the authors.

---
