# On the Effectiveness of Prompt Stealing Attacks on In-The-Wild Prompts
This is the official repository of the IEEE S&P 2025 paper [**On the Effectiveness of Prompt Stealing Attacks on In-The-Wild Prompts**](https://www.computer.org/csdl/proceedings-article/sp/2025/223600a355/26hiTFMb8eQ).

**All the following updates will be released here first in the future.**

## How to use this repository?
### Install and set the ENV
1. Clone this repository.
2. Prepare the python ENV.
```
conda create -n T-GPS python=3.10
conda activate T-GPS
cd PATH_TO_THE_REPOSITORY
pip install -r requirements.txt
```

### In the PromptRecovery\src\gradient_utils.py
```
OPENAI_API_KEY = 'your_openai_api_key_here' 
```

### Run the gradient_attack.py

```
python gradient_attack.py --max_rounds 6 --max_workers 16 --beam_size 4 --model 'gpt-3.5-turbo-0125' --round 'r1' --data_path 'PromptRecovery/data/in_the_wild_prompts/in_the_wild_filter_prompt2output_test.json' --save_dict_path 'results' --save_log_path 'logs/attack_log.md'
```

### Parameter Breakdown

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `--max_rounds` | 6 | Sets the maximum number of optimization iterations the algorithm will perform when attempting to recover prompts. Each round refines the candidate prompts further. |
| `--max_workers` | 16 | Defines the number of parallel workers/threads to use during processing, enabling concurrent API calls to speed up the attack process. |
| `--beam_size` | 4 | Controls how many top candidate prompts to keep at each step of the optimization (beam search). Higher values explore more possibilities but increase computational cost. |
| `--model` | 'gpt-3.5-turbo-0125' | Specifies which OpenAI model to target for the prompt stealing attack.|
| `--round` | 'r1' | A label for the repeating experimental run, useful for organizing outputs from different repeating attack attempts.  |
| `--data_path` | 'PromptRecovery/data/in_the_wild_prompts/in_the_wild_filter_prompt2output_test.json' | The directory containing target prompt-response pairs to attempt recovery on. We list In-The-Wild, Awesome-GPT-Prompts, and the defense examples in the data folder.|
| `--save_dict_path` | 'results' | Directory where the recovered prompts will be saved (in JSON format). |
| `--save_log_path` | 'logs/attack_log.md' | File path for the attack's execution log, saved in Markdown format with timestamps as configured in the setup_logger function. |


## TO DO

- [ ] Prompt Analysis.
- [ ] Evaluation Process.
- [ ] Implementation of Existing Attacks.
