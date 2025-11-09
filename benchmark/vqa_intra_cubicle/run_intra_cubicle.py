import os
import json
import argparse
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from utils import setup_logger, generate_questions, save_results, load_questions

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
        "num_frames": 17,  # Use -1 for all frames, or specify a number (it will be uniformly sampled)
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

    questions_file = "data/questions/global_changes/Object_State_Questions.json"
    

    model_provider = "gpt4o"

    log_filename = setup_logger(questions_file)
    logging.info("Logger initialized")
    logging.info(f"Using model: {model_provider}")
    logging.info(f"Using questions file: {questions_file}")

    
    # Initialize the model
    model_config = MODEL_CONFIGS.get(model_provider, MODEL_CONFIGS[MODEL_PROVIDER])
    model = create_model(model_provider, **model_config)

    questions_dict = load_questions(questions_file)
    logging.info(f"Loaded {len(questions_dict)} questions from {questions_file}")

    initial_video = 0
    final_video = questions_dict[list(questions_dict.keys())[-1]]["Final Video"]
    logging.info(f"Processing videos from {initial_video} to {final_video}")

    dict_answers = {}
    for i in range(initial_video, final_video):
        logging.info(f"Processing video pair {i} and {i+1}")


        episode_path_1 = f"data/global_changes_videos/episode_{i}_720p_10fps.mp4"
        episode_path_2 = f"data/global_changes_videos/episode_{i+1}_720p_10fps.mp4"
        assert os.path.exists(episode_path_1) and os.path.exists(episode_path_2), "Missing file:" + episode_path_1 + " or " + episode_path_2


        prompt = generate_questions(questions_dict, i)
        logging.info("Generated questions prompt")
        logging.info(f"Prompt being sent to model :\n{prompt}")
        # Generate response using the model interface
        logging.info(f"Processing episode {i}...")
        resp_text = model.generate_response(episode_path_1, episode_path_2, prompt, AnswersResponse)

        logging.info(f"Received response from {model.get_model_name()}")
        logging.info(f"Raw response from {model.get_model_name()}: {resp_text}")  


        if not resp_text:
            logging.error(f"Empty response received from {model.get_model_name()}")
            raise ValueError(f"Empty response from {model.get_model_name()}")
        try:
            answers_json = json.loads(resp_text)
            logging.info(f"Successfully parsed JSON response: {answers_json}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            logging.error(f"Raw response that failed to parse: {resp_text}")
            raise

        for answer in answers_json['answers']:
            dict_answers[answer['question']] = answer['answer']
    

    logging.info("Completed processing all video pairs")
    logging.info(f"Final answers dictionary: {dict_answers}")

    total_correct = 0
    total_questions = 0

    for question in questions_dict:
        total_questions += 1
        if questions_dict[question]["Correct Choice"] == dict_answers[question]:
            total_correct += 1

    logging.info(f"Scoring complete: {total_correct} correct out of {total_questions} questions")

    # Collect video paths
    # video_paths = []
    # for i in range(initial_video, final_video):
    #     video_paths.append(VIDEO_PATH_PATTERN.format(i=i))
    #     video_paths.append(VIDEO_PATH_PATTERN.format(i=i+1))
    # logging.info(f"Collected {len(video_paths)} video paths")

    # Save all results
    save_results(
        questions_file=questions_file,
        prompt=prompt,
        questions_dict=questions_dict,
        dict_answers=dict_answers,
        total_correct=total_correct,
        total_questions=total_questions,
        model_name=model_provider,
        temperature=MODEL_CONFIGS[model_provider]["temperature"]
    )
    logging.info("Benchmark execution completed successfully")