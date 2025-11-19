import os
import json
import argparse
from typing import List, Dict
import time
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from utils import setup_logger, generate_question, save_results_single_question, load_questions, load_results

from model_interface import create_model

load_dotenv()


# --- Model Configuration ---
# Supported models: "gemini", "gpt4o"
MODEL_PROVIDER = "gemini"

# Provider-specific model names
MODEL_CONFIGS = {
    "gemini": {
        "model_name": "gemini-2.5-pro",
        "temperature": 0.0,
    },
    "gpt4o": {
        "model_name": "gpt-4o",
        "temperature": 0.0,
        "num_frames": 35,  # Use -1 for all frames, or specify a number (it will be uniformly sampled)
    }
}





# --- Pydantic schemas to structure VLM output ---
class AnswerItem(BaseModel):
    question: str  # Question identifier (e.g., "Q1", "Q2", etc.)
    answer: str    # Single letter answer (e.g., "A", "B", "C", "D", "E")

class AnswersResponse(BaseModel):
    answers: List[AnswerItem]  # List of question-answer pairs


# ---------------------------
# Main loop: prompting & evaluation
# ---------------------------
if __name__ == "__main__":

    questions_file = "data\questions\global_changes\Object_State_intra_cubcile_sp.json"
    
    resume_from = "results/vqa_intra_cubical_spatial_association_single_question/gpt4o/Object_State_intra_cubcile_sp_20251110_173233_results.json"

    model_provider = "gpt4o"

    if resume_from is not None:
        results = load_results(resume_from)

    log_filename = setup_logger(questions_file)
    logging.info("Logger initialized")
    logging.info(f"Using model: {model_provider}")
    logging.info(f"Using questions file: {questions_file}")

    
    # Initialize the model
    model_config = MODEL_CONFIGS.get(model_provider, MODEL_CONFIGS[MODEL_PROVIDER])
    model = create_model(model_provider, **model_config)

    questions_dict = load_questions(questions_file)
    logging.info(f"Loaded {len(questions_dict)} questions from {questions_file}")

    if resume_from is not None:
        dict_answers = results["answers"]
        # Get list of already processed questions
        processed_questions = set(dict_answers.keys())
        logging.info(f"Resuming from previous run. Already processed {len(processed_questions)} questions.")
    else:
        dict_answers = {}
        processed_questions = set()

    status = "completed"
    for question in questions_dict:
        # Skip already processed questions
        if question in processed_questions:
            logging.info(f"Skipping already processed question {question}")
            continue
            
        video_number = questions_dict[question]["Video"]    
        logging.info(f"Processing Question {question} for video {video_number}")



        episode_path_1 = f"data/global_changes_videos/episode_{video_number}_720p_10fps.mp4"
        assert os.path.exists(episode_path_1), "Missing file:" + episode_path_1

        prompt = generate_question(questions_dict, question)
        logging.info("Generated questions prompt")
        logging.info(f"Prompt being sent to model :\n{prompt}")
        # Generate response using the model interface
        logging.info(f"Processing question {question}...")
        resp_text = model.generate_response(episode_path_1, prompt, AnswersResponse)

        logging.info(f"Received response from {model.get_model_name()}")
        logging.info(f"Raw response from {model.get_model_name()}: {resp_text}")  


        if not resp_text:
            logging.error(f"Empty response received from {model.get_model_name()}")
            status = "not completed"
            break
        try:
            answers_json = json.loads(resp_text)
            logging.info(f"Successfully parsed JSON response: {answers_json}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            logging.error(f"Raw response that failed to parse: {resp_text}")
            status = "not completed"
            break

        for answer in answers_json['answers']:
            dict_answers[answer['question']] = answer['answer']
        logging.info(f"Sleeping for 10 seconds")
        time.sleep(10)
    

    logging.info("Completed processing all questions")
    logging.info(f"Final answers dictionary: {dict_answers}")
    

    total_correct = 0
    total_questions = 0

    if status == "completed":
        for question in questions_dict:
            total_questions += 1

            if questions_dict[question]["Correct Choice"] == dict_answers[question]:
                total_correct += 1

        logging.info(f"Scoring complete: {total_correct} correct out of {total_questions} questions")

    # Save all results
    save_results_single_question(
        questions_file=questions_file,
        prompt=prompt,
        questions_dict=questions_dict,
        dict_answers=dict_answers,
        total_correct=total_correct,
        total_questions=total_questions,
        model_name=model_provider,
        temperature=MODEL_CONFIGS[model_provider]["temperature"],
        status=status
    )
    logging.info("Benchmark execution completed successfully")