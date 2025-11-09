
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
    
    # Configure logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Overwrite existing log file
    )
    
    # Add console handler to see logs in terminal too
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Starting video understanding benchmark for {questions_file}")
    return log_filename


def generate_questions(Q_dict,video_number):
    logging.info(f"Generating questions for video number {video_number}")
    prompt_start = """

You are taking a multiple-choice benchmark.  
  
To answer the questions, you will be provided with two videos. The first video is the initial state of the scene, and the second video is the final state of the scene.
Please pay attention to the changes of objects in the scene between the two videos.
Furthermore, if we say an object had been removed, vanished, or disappeared from a cubicle it means it will not be present in the second video.
If we say an object has appeared in a cubicle it means it will be present in the second video.
If we say an object has been moved, it means it has moved within the cubicle or between cubicles.

Please answer the question to best of your ability. 

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
    questions_list = [key for key, value in Q_dict.items() if value["Initial Video"] == video_number]
    logging.info(f"Found {len(questions_list)} questions for video {video_number}")

    questions_prompt_list = []
    for i, question in enumerate(questions_list):
        questions_prompt_list.append(question + ":\n" + "Question: " + Q_dict[question]["Question"] + "\n")
        for choice in list(Q_dict[question]["Multiple Choice"].keys()):
            questions_prompt_list[i] += choice + ": " + Q_dict[question]["Multiple Choice"][choice] + "\n"

    prompt_end = """
"""

    questions_prompt = prompt_start + "\n".join(questions_prompt_list) + "\n" + prompt_end
    logging.debug(f"Generated prompt: {questions_prompt}")
    return questions_prompt


def save_results(questions_file, prompt, questions_dict, dict_answers, total_correct, total_questions, model_name, temperature):
    logging.info(f"Saving results for {questions_file}")
    # Create output filename based on input filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.splitext(os.path.basename(questions_file))[0]  + f"_{timestamp}_results.json"
    
    output_dir = f"results/vqa_intra_cubicle/{model_name}"

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
