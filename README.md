# 🧠 AI-Powered Prediction of Drug-Target Interaction Potency Using Molecular Representations

This repository contains code, datasets, and models for predicting **Drug-Target Interaction (DTI) potency** using both traditional machine learning methods and **Graph Neural Networks (GNNs)**. The primary goal is to predict **pIC50** values from molecular structure data using representations derived from SMILES strings and molecular graphs.

---

## 📘 Abstract

Drug discovery relies heavily on accurately predicting the strength of interactions between small molecules and their biological targets. This project explores:

- A **baseline Random Forest model** trained on traditional molecular descriptors
- Advanced **Graph Neural Network models** (GCN, GIN) trained on molecular graphs

The results demonstrate that GNN-based models significantly outperform traditional methods in predicting bioactivity (pIC50).

---

## 📁 Repository Structure

```
.
├── data/                     # Scripts for data download and preprocessing
├── models/                  
│   ├── baseline_rf_model.ipynb    # Random Forest baseline notebook
│   ├── gnn_gcn_model.py           # GCN model implementation
│   ├── gnn_gin_model.py           # GIN model implementation
│   └── train_gnn.py               # GNN training loop and evaluation
├── results/                  # Plots and evaluation metrics
├── utils/                    # Helper functions
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview
└── config.yaml               # Configuration file for experiments
```

---

## 🔬 Methods

### 1. Dataset and Preprocessing

- **Source:** [ChEMBL](https://www.ebi.ac.uk/chembl/)
- **Filters:**
  - Bioactivity type: `'IC50'`
  - Standard units: `'nM'`
  - Valid SMILES and activity values only
- **Transformation:**
  ```
  pIC50 = -log10(IC50 * 10⁻⁹ M)
  ```

### 2. Feature Extraction

#### 🔹 Baseline Model Features (via RDKit):

- Molecular Weight
- LogP
- Hydrogen Bond Donors / Acceptors
- Topological Polar Surface Area (TPSA)
- Rotatable Bonds

#### 🔹 GNN Model Input:

- Molecular graphs constructed from SMILES:
  - Atoms → Nodes
  - Bonds → Edges
- Node features: Atom type, degree, formal charge, hybridization, etc.
- Graph construction handled via `PyTorch Geometric`

---

## 🤖 Models

### 🟩 1. Baseline: Random Forest Regressor

- **Library:** `scikit-learn`
- **Input:** Molecular descriptors (handcrafted)
- **Performance:**
  - RMSE: `1.324`
  - R²: `0.254`

### 🟦 2. Graph Neural Networks

Implemented using **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)**.

#### 🔸 Graph Convolutional Network (GCN)

- Aggregates information from neighboring atoms
- Suitable for learning local structural patterns

#### 🔸 Graph Isomorphism Network (GIN)

- Higher expressiveness
- Capable of distinguishing subtle molecular differences (e.g., stereochemistry)

**GNN Model Features:**
- Multi-layer GNNs with batch normalization and ReLU
- Global pooling (mean or attention-based)
- Fully connected layers for final pIC50 prediction

---

## 📊 Results Summary

| Model       | RMSE   | R² Score |
|-------------|--------|----------|
| Random Forest | 1.324 | 0.254    |
| GCN          | 0.934 | 0.618    |
| GIN          | 0.892 | 0.664    |

> GNN-based models significantly outperformed traditional methods, showing stronger structure-activity learning capabilities.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/dreamboat26/AI-Powered-Prediction-of-Drug-Target-Interaction-Potency-using-Molecular-Representations.git
cd AI-Powered-Prediction-of-Drug-Target-Interaction-Potency-using-Molecular-Representations
```

### 2. Set up Python environment

We recommend Python 3.9+ and a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### Additional Notes:
- PyTorch and PyTorch Geometric may require specific versions based on CUDA. See [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

---

## 🧪 Usage

### Run Baseline Random Forest

```bash
jupyter notebook models/baseline_rf_model.ipynb
```

### Train GNN Model (GCN or GIN)

```bash
python models/train_gnn.py --model gcn  
```

---

## 📚 References

1. Gaulton et al. (2012). ChEMBL: a large-scale bioactivity database.
2. Wu et al. (2018). MoleculeNet: A Benchmark for Molecular Machine Learning.
3. Kipf & Welling (2016). Semi-Supervised Classification with Graph Convolutional Networks.
4. Xu et al. (2019). How Powerful Are Graph Neural Networks?

---

## 👩‍🔬 Authors

- Mahule – [@dreamboat26](https://github.com/dreamboat26)  
  *Contributions: Data engineering, model implementation (RF, GCN, GIN), results analysis, manuscript writing.*

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

Thanks to the developers of [ChEMBL](https://www.ebi.ac.uk/chembl/), [RDKit](https://www.rdkit.org/), [scikit-learn](https://scikit-learn.org/), [PyTorch](https://pytorch.org/), and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for providing open-source tools that made this project possible.
