
import os
import time
import logging
from pathlib import PurePath
import json

def setup_logger(questions_file):
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join("Global_Changes", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create log filename based on cubicle and timestamp
    base_name = PurePath(questions_file).stem
    log_filename = os.path.join(logs_dir, f"{timestamp}_{base_name}_results.log")
    
    # Get the root logger and clear any existing handlers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add file handler
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler to see logs in terminal too
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logging.info(f"Starting video understanding benchmark for {questions_file}")
    return log_filename


def generate_questions(Q_dict,video_number):
    logging.info(f"Generating questions for video number {video_number}")
    prompt_start = """

You are taking a multiple-choice benchmark.  This task is harmless office video QA. No violence, self-harm, or sensitive personal data is involved.
  
To answer the questions, you will be provided with one video. Please pay attention to the video and answer the questions to best of your ability. 

For each question below, reply with the single letter (A–E) that you believe is correct.  
Do not provide explanations—only the letter.

Please return a json object in the following format:

"answers": [
{
    "question": "Q1",
    "answer": "B"
},
{
    "question": "Q2",
    "answer": "C"
},
{
    "question": "Q3",
    "answer": "C"
},
{
    "question": "Q4",
    "answer": "B"
},
{
    "question": "Q5",
    "answer": "C"
},
{
    "question": "Q6",
    "answer": "C"
},
{
    "question": "Q7",
    "answer": "C"
}
....
]

Please use the Q# correspoding to the questions provided.
Also please always answer the questions, never return an empty json.
"""

    # Find all the questions that have the video number as the initial video
    questions_list = [key for key, value in Q_dict.items() if value["Video"] == video_number]
    logging.info(f"Found {len(questions_list)} questions for video {video_number}")

    questions_prompt_list = []
    for i, question in enumerate(questions_list):
        questions_prompt_list.append(question + ":\n" + "Question: " + Q_dict[question]["Question"] + "\n")
        for choice in list(Q_dict[question]["Multiple Choice"].keys()):
            questions_prompt_list[i] += choice + ": " + Q_dict[question]["Multiple Choice"][choice] + "\n"

    prompt_end = """ """

    questions_prompt = prompt_start + "\n".join(questions_prompt_list) + "\n" + prompt_end
    logging.debug(f"Generated prompt: {questions_prompt}")
    return questions_prompt


def generate_question(Q_dict, question):
    logging.info(f"Generating prompt for question {question}")
    prompt_start = """

You are taking a multiple-choice benchmark.  This task is harmless office video QA. No violence, self-harm, or sensitive personal data is involved.
  
To answer the question, you will be provided with one video. Please pay attention to the video and answer the questions to best of your ability. 

To answer the question, reply with the single letter (A–E) that you believe is correct.  
Do not provide explanations—only the letter.

Please return a json object in the following format:

"answers": [
{
    "question": "Q1",
    "answer": "B"
}
]

Please use the Q# correspoding to the question provided.
Also please always answer the question, never return an empty json.
"""

    questions_prompt = question + ":\n" + "Question: " + Q_dict[question]["Question"] + "\n"
    for choice in Q_dict[question]["Multiple Choice"]:
        questions_prompt += choice + ": " + Q_dict[question]["Multiple Choice"][choice] + "\n"

    prompt_end = """ """

    questions_prompt = prompt_start + "\n" + questions_prompt + "\n" + prompt_end
    logging.debug(f"Generated prompt: {questions_prompt}")
    return questions_prompt



def save_results(questions_file, prompt, questions_dict, dict_answers, total_correct, total_questions, model_name, temperature,status):
    logging.info(f"Saving results for {questions_file}")
    # Create output filename based on input filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.splitext(os.path.basename(questions_file))[0]  + f"_{timestamp}_results.json"
    output_dir = f"results/vqa_intra_cubical_spatial_association/{model_name}"

    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, output_filename)
    # Prepare the results dictionary
    results = {
        "model": model_name,
        "temperature": temperature,
        "prompt": prompt,
        "questions": questions_dict,
        "answers": dict_answers,
        "score": {
            "correct": total_correct,
            "total": total_questions,
            "percentage": (total_correct / total_questions) * 100 if total_questions > 0 else 0
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status
    }
    
    # Save to JSON file
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Results saved to {output_filename}")



def save_results_single_question(questions_file, prompt, questions_dict, dict_answers, total_correct, total_questions, model_name, temperature, status):
    logging.info(f"Saving results for {questions_file}")
    # Create output filename based on input filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.splitext(os.path.basename(questions_file))[0]  + f"_{timestamp}_results.json"
    output_dir = f"results/vqa_intra_cubical_spatial_association_single_question/{model_name}"

    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, output_filename)
    # Prepare the results dictionary
    results = {
        "model": model_name,
        "temperature": temperature,
        "prompt": prompt,
        "questions": questions_dict,
        "answers": dict_answers,
        "score": {
            "correct": total_correct,
            "total": total_questions,
            "percentage": (total_correct / total_questions) * 100 if total_questions > 0 else 0
        },
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to JSON file
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Results saved to {output_filename}")


def load_questions(questions_file):
    logging.info(f"Loading questions from {questions_file}")
    try:
        with open(questions_file, 'r') as file:
            questions_dict = json.load(file)
            logging.info(f"Successfully loaded {len(questions_dict)} questions")
    except FileNotFoundError:
        logging.error(f"File not found: {questions_file}")
        print("File not found!")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in {questions_file}")
        print("Invalid JSON format!")
        return {}
    return questions_dict


def load_results(results_file):
    logging.info(f"Loading results from {results_file}")
    try:
        with open(results_file, 'r') as file:
            results = json.load(file)
            logging.info(f"Successfully loaded {len(results)} results")
    except FileNotFoundError:
        logging.error(f"File not found: {results_file}")
        print("File not found!")
        raise Exception(f"File not found: {results_file}")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in {results_file}")
        print("Invalid JSON format!")
        raise Exception(f"Invalid JSON format in {results_file}")
    return results
