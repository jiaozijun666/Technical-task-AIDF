# Technical-task-AIDF
Pre-project Task - Hallucination Detection

All files run on a NVIDIA A100 GPU rented from vest.ai with model LlaMa-3.1-8B requested from Meta official website.
### Repository Structure
```
Technical-task-AIDF/
│
├── baseline/                                 
│   ├── baseline.py     
│   ├── utils.py                              
│   └── __init__.py
│
├── HaMI/                                     
│   ├── hami.py                               
│   ├── enhanced_hami.py                     
│   └── __init__.py
│
├── src/                                      
│   ├── api.py                                
│   ├── data.py                               
│   ├── final_select.py                       
│   ├── model.py                             
│   ├── multi_sample.py                       # Multi-answer generation using LLM
│   ├── process_data.py                       # Preprocess and split dataset
│   ├── prompt.py                             # Prompt templates for baseline generation
│   ├── random_pairs.py                       # Construct random QA evaluation pairs
│   ├── refined_set.py                        # Final dataset refinement
│   └── __init__.py
│
├── main.py                                   # General entry point for full pipeline
├── requirements.txt                          # Dependency list
├── LICENSE                                   # MIT License
├── .gitignore                                # Ignore cache, models, and large outputs
└── README.md                                 # Project documentation
```
### Detailed process as follows:
Clone the repository to Google Colab
```{bash}
git clone https://github.com/jiaozijun666/Technical-task-AIDF.git
cd Technical-task-AIDF
```
Creat virtual environment
```{python}
python3 -m venv .venv
source .venv/bin/activate
```
Download the model Llama-3.1-8B to environment
```{bash}
# Download ModelScope
source /workspace/Technical-tesk-AIDF/.venv/bin/active 2>/dev/null || true
pip install -U modelscope transoformers accelerate bitsandbytes sentencepiece

# Transformer mode
python - <<'PY'
from modelscope import snapshot_download
d = snapshot_download(
    'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    cache_dir='/workspace/models',
    revision=None
)
print(d)
PY

# Check status
ls -lh /workspace/models/LLM-Research/Meta-Llama-3.1-8B-Instruct | head
du -sh /worksoace/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
```
输入openai api key
```{bash}
pip install --upgrade openai
export OPENAI_API_KEY="sk-xxxxxxxx"
```

Downloading the necessary packages
```{bash}
pip install -r requirements.txt
```

Running the piplines in following order
```{python}
python src/process_data.py
python src/multi_sample.py
python src/final_select.py
python src/random_pairs.py
python src/refined_set.py
python main.py  
```
