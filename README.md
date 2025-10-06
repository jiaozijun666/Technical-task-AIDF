# Technical-task-AIDF
Pre-project Task - Hallucination Detection

```{python}
!git clone https://github.com/jiaozijun666/Technical-task-AIDF.git
%cd Technical-task-AIDF
```
```{python}
!pip install -r requirements.txt
!pip install bitsandbytes accelerate transformers datasets
```
```{python}
from huggingface_hub import login
login()
```
```{python}
import os, sys
sys.path.append('/content/Technical-task-AIDF/src')
os.chdir('/content/Technical-task-AIDF')
```
```{python}
!python src/process_data.py
!python src/multi_sample.py
```
