import os
import argparse
import logging
import datetime
import concurrent.futures

from vllm import LLM
from src.gradient_utils import *
from tqdm import tqdm
from collections import defaultdict
from collections import OrderedDict
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_rounds', default=6, type=int)
    parser.add_argument('--max_workers', default=16, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--model', default='gpt-3.5-turbo-0125')
    parser.add_argument('--round', default='r1')
    parser.add_argument('--data_path', default='PromptRecovery/selected_data/defense_prompts/example_prompts2outputs.json')
    parser.add_argument('--save_dict_path', default='')
    parser.add_argument('--save_log_path', default='')
    parser.add_argument('--start_index', default=0, type=int)    
    parser.add_argument('--end_index',default=2, type=int)   
    args = parser.parse_args()

    return args

def save_dict(dict, filename='target_prompt2recovery_prompt.json'):
    with open(filename, 'w') as file:
        json.dump(dict, file, indent=4) 

def setup_logger(name='optimizer'):
    # Creating a logger name with the current time
    logger_name = f"{name}"
    logger = logging.getLogger(logger_name)

    # Setup logging to write into a Markdown file with the timestamp
    log_filename = f'{logger_name}.md'
    file_handler = logging.FileHandler(log_filename, mode='w')
    
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    return logger


Test = False
if __name__ == '__main__':

    args = get_args()
    total_target_data = load_responses_from_file(args.data_path)
    total_target_prompts = [total_target_prompt for total_target_prompt, _ in total_target_data.items()]
    total_target_outputs = total_target_data
    start_index = args.start_index
    end_index = args.end_index
    end_index = min(args.end_index, len(total_target_prompts))
    model = args.model
    experiment_round = args.round

    for cur_start_index in range(start_index, end_index, 50):
        cur_end_index = min(cur_start_index+50, len(total_target_prompts))
        save_dict_path = f'{args.save_path}/{experiment_round}/{cur_start_index}_{cur_end_index}.json'
        save_log_path = f'{args.log_path}/{experiment_round}/log/{cur_start_index}_{cur_end_index}'
        if Test:
            save_log_path = f'PromptRecovery/logs/test'
        print("Save_dict_path: ",save_dict_path)
        print('Save_log_path: ',save_log_path)
        print(cur_start_index, cur_end_index)
        target_prompts = total_target_prompts[cur_start_index:cur_end_index]
        target_outputs = {target_prompt:total_target_outputs[target_prompt] for target_prompt in target_prompts}
        assert len(target_outputs) == len(target_prompts), "List lengths are not equal"
    
        optimizer_logger = setup_logger(save_log_path)

        print(f'target_prompts length:{len(target_prompts)}')
        print(f'target_outputs length:{len(target_outputs)}')

        TextGradient_target_prompt2recovery_prompts = OrderedDict()
        target_prompts2best_scores = {target_prompt:0 for target_prompt in target_prompts}
        target_outputs2target_prompt = {target_outputs[target_prompt]:target_prompt for target_prompt in target_prompts}
        
        # Extract Key Info: Purpose, Key words, Structure Format, Tone and Style
        target_outputs2draft_prompts = multi_get_draft_prompts(target_outputs)

        print(f'target_output2draft_prompts length:{len(target_outputs2draft_prompts)}')
        target_prompts2candidates = {target_prompt:[target_outputs2draft_prompts[(target_prompt,target_outputs[target_prompt])]] for target_prompt in target_prompts}
        for round in tqdm(range(args.max_rounds+1)):
            target_prompts2new_candidates = OrderedDict()
            if round > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                    # expand_candidates(target_prompt, candidate_prompts, target_output):
                    futures = [executor.submit(expand_candidates, target_prompt,  target_prompts2candidates[target_prompt], target_outputs[target_prompt], model=model) for target_prompt in target_prompts]
                    for future in futures:
                        target_prompt, new_candidate_prompts, logs = future.result()
                        optimizer_logger.info(logs)
                        target_prompts2new_candidates[target_prompt] = new_candidate_prompts

            if round == 0:
                target_prompts2new_candidates = target_prompts2candidates
            optimizer_logger.info('-----------------------END OF GENERATING NEW CANDIDATES------------------------ \n')
            target_prompts2candidates_scores = score_new_candidates(target_prompts, target_outputs, target_prompts2new_candidates)
            optimizer_logger.info('-----------------------START OF RANKING ACCORDING TO SCORE--------------------------\n')
            for target_prompt, candidates2scores in target_prompts2candidates_scores.items():
                sorted_candidates_scores = sorted(candidates2scores.items(), key=lambda item: item[1], reverse=True)
                new_candidates_scores = sorted_candidates_scores[:args.beam_size]
                candidates = [ candidate_score[0] for candidate_score in new_candidates_scores]
                # Set the target_prompt with the new candidates after graident descent
                target_prompts2candidates[target_prompt] = candidates
                optimizer_logger.info(f'target_prompt:{target_prompt} new_candidates2scores:{new_candidates_scores} \n')
                if new_candidates_scores[0][1] >= target_prompts2best_scores[target_prompt]:
                    target_prompts2best_scores[target_prompt] = new_candidates_scores[0][1]
                    TextGradient_target_prompt2recovery_prompts[target_prompt] = new_candidates_scores[0][0]
        optimizer_logger.info('Successfully complete the task!')
        save_dict(TextGradient_target_prompt2recovery_prompts, filename=save_dict_path)  
        optimizer_logger.info(f'save to{save_dict_path}')

