# Technical-task-AIDF
Pre-project Task - Hallucination Detection

All files run on a NVIDIA A100 GPU rented from vest.ai with model LLaMA-3.1-8B requested from Meta official website.

### Repository Structure
```
Technical-task-AIDF/
│
├── baseline/                                 # Baseline implementations
│   ├── internal_representation_based/        # Internal representation-based baselines
│   │   ├── ccs.py                            # Centered Cosine Similarity
│   │   ├── haloscope.py                      # Haloscope baseline
│   │   ├── saplma.py                         # SAPLMA baseline
│   │   └── __init__.py
│   │
│   ├── uncertainty_based/                    # Uncertainty-based baselines
│   │   ├── mars.py                           # MARS (Monte Carlo sampling)
│   │   ├── mars_se.py                        # MARS-SE (Semantic Entropy)
│   │   ├── p_true.py                         # True probability estimation
│   │   ├── perplexity.py                     # Perplexity baseline
│   │   ├── semantic_entropy.py               # Semantic Entropy baseline
│   │   └── __init__.py
│   │
│   ├── utils.py                              # Common baseline utilities
│   └── __init__.py
│
├── HaMI/                                     # Proposed HaMI and enhanced HaMI*
│   ├── hami.py                               # Original HaMI implementation
│   ├── enhanced_hami.py                      # Improved HaMI* with semantic reasoning
│   └── __init__.py
│
├── src/                                      # Core pipeline and data processing scripts
│   ├── api.py                                # Unified API for model calls (HF/Ollama)
│   ├── data.py                               # Dataset and utility functions
│   ├── final_select.py                       # Select high-quality samples
│   ├── model.py                              # Model loader and generation config
│   ├── multi_sample.py                       # Multi-answer generation using LLM
│   ├── process_data.py                       # Preprocess and split dataset
│   ├── prompt.py                             # Prompt templates for baseline generation
│   ├── random_pairs.py                       # Construct random QA evaluation pairs
│   ├── refined_set.py                        # Final dataset refinement
│   └── __init__.py
│
├── test/                                     # Evaluation and debugging scripts
│   ├── multi_sample_test5.py                 # Lightweight version of multi-sample (small-scale)
│   ├── final_select_test5.py                 # Light version of final selection
│   ├── refined_set_test5.py                  # Light version of refined set generation
│   └── main_test.py                          # Evaluate 4 baselines: Perplexity / Entropy / HaMI / HaMI*
│
├── data/ (generated)                         # Generated datasets (after pipeline execution)
│   ├── squad_train.json
│   ├── squad_test.json
│   ├── squad_final.json
│   ├── squad_random_pairs.json
│   └── squad_multi_debug.json
│
├── results/ (generated)                      # Evaluation outputs and metrics
│   ├── final_results.json                    # Predictions from all baselines and HaMI variants
│   └── summary_5metrics.json                 # Computed AUROC / Accuracy / F1 scores
│
├── main.py                                   # General entry point for full pipeline
├── requirements.txt                          # Dependency list
├── LICENSE                                   # MIT License
├── .gitignore                                # Ignore cache, models, and large outputs
└── README.md                                 # Project documentation
```
### Detailed process as follows:
Clone the repository to Google Colab
```{python}
git clone https://github.com/jiaozijun666/Technical-task-AIDF.git
cd Technical-task-AIDF
```
Downloading the necessary packages
```{python}
pip install -r requirements.txt
pip install bitsandbytes accelerate transformers datasets
```
