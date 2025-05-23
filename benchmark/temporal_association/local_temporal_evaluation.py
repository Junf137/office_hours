import os
import re
import glob
import base64
import json
import time
import multiprocessing
from typing import List, Tuple, Dict
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# Constants
VIDEOS_ROOT_DIR = "./data/local_changes_videos"
CSV_ROOT_DIR = "./data/csv_files"
PROMPT_TEMPLATE_PATH = "./prompt/Local Temporal.md"
EVA_PROMPT = "./prompt/Local Temporal Evaluation.md"
OUTPUT_DIR = "./output/temporal_association"
GEMINI_MODEL = "models/gemini-2.5-pro-preview-05-06"
MAX_PROC = 10


def unify_text(text: str) -> str:
    """Unify text by removing extra spaces."""
    return re.sub(r"\s+", " ", text).strip()


def load_api_key(key_path: str) -> str:
    """Load the Gemini API key from a file."""
    try:
        with open(key_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at {key_path}")


def load_prompt_template(template_path: str) -> str:
    """Load the prompt template from a file."""
    try:
        with open(template_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template file not found at {template_path}")


def get_cubicle_video_pairs(videos_root_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    Find all cubicle folders and their video files matching the pattern ep_{episode_number}_a.{ext}

    Returns:
        Dict mapping cubicle names to lists of (episode_number, video_path) tuples
    """
    cubicle_videos = {}

    # Ensure the root directory exists
    if not os.path.exists(videos_root_dir):
        raise FileNotFoundError(f"Videos root directory not found at {videos_root_dir}")

    # Find all cubicle folders
    for cubicle_dir in glob.glob(os.path.join(videos_root_dir, "*")):
        if not os.path.isdir(cubicle_dir):
            continue

        cubicle_name = os.path.basename(cubicle_dir)
        videos = []

        # Find all videos matching the pattern ep_{episode_number}_a.{ext}
        for video_path in glob.glob(os.path.join(cubicle_dir, "ep_*_a.*")):
            # Extract episode number using regex
            match = re.search(r"ep_(\d+)_a\.", video_path)
            if match:
                episode_number = int(match.group(1))
                videos.append((episode_number, video_path))

        # Sort videos by episode number
        videos.sort(key=lambda x: x[0])

        if videos:
            cubicle_videos[cubicle_name] = videos

    return cubicle_videos


def create_video_pairs(videos: List[Tuple[int, str]]) -> List[Tuple[Tuple[int, str], Tuple[int, str]]]:
    """
    Create pairs of consecutive videos.

    Args:
        videos: List of (episode_number, video_path) tuples

    Returns:
        List of ((ep1_num, ep1_path), (ep2_num, ep2_path)) pairs
    """
    pairs = []
    for i in range(len(videos) - 1):
        pairs.append((videos[i], videos[i + 1]))
    return pairs


def prepare_prompt(template: str, cubicle_name: str, episode1_num: int, episode2_num: int) -> str:
    """
    Prepare a prompt by replacing placeholders in the template.

    Args:
        template: The prompt template string
        cubicle_name: Name of the cubicle
        episode1_num: First episode number
        episode2_num: Second episode number

    Returns:
        Prepared prompt string
    """
    episode_name_1 = f"ep_{episode1_num}_a"
    episode_name_2 = f"ep_{episode2_num}_a"

    # Replace placeholders
    prompt = template.replace("[cubicle_name]", cubicle_name)
    prompt = prompt.replace("[episode_name_1]", episode_name_1)
    prompt = prompt.replace("[episode_name_2]", episode_name_2)

    return prompt


def encode_video(video_path: str) -> str:
    """
    Encode a video file as base64.

    Args:
        video_path: Path to the video file

    Returns:
        Base64-encoded video string
    """
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def gemini_response(
    api_key: str,
    prompt: str,
    ep1_path: str,
    ep2_path: str,
) -> str:
    """
    Generate a response from Gemini model using the provided API key, prompt, and videos.

    Args:
        api_key: The Gemini API key
        prompt: The text prompt to send
        video_paths: List of paths to video files to include

    Returns:
        Generated text response
    """
    client = genai.Client(api_key=api_key)

    ep1_file = client.files.upload(file=ep1_path)
    ep2_file = client.files.upload(file=ep2_path)

    # Wait until the file is processed
    while not ep1_file.state or ep1_file.state.name != "ACTIVE":
        print("File state:", ep1_file.state)
        time.sleep(5)
        ep1_file = client.files.get(name=ep1_file.name)

    while not ep2_file.state or ep2_file.state.name != "ACTIVE":
        print("File state:", ep2_file.state)
        time.sleep(5)
        ep2_file = client.files.get(name=ep2_file.name)

    # Generate content
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[ep1_file, ep2_file, prompt],
            config=types.GenerateContentConfig(
                temperature=1.0,
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating content: {e}")
        return ""


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON content from a response text.

    Args:
        response: Response text containing JSON content inside code blocks

    Returns:
        Extracted JSON content
    """
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
    if json_match:
        return json_match.group(1).strip()
    else:
        # Check if the response is directly a JSON without code block markers
        try:
            # Validate if the response is JSON
            json.loads(response)
            return response
        except json.JSONDecodeError:
            return "{}"


def save_json_response(json_content: str, output_path: str) -> None:
    """
    Save JSON content to a file.

    Args:
        json_content: JSON content to save
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as json_file:
        json_file.write(json_content + "\n")


def process_single_cubicle(args: Tuple[str, List[Tuple[int, str]], str, str]) -> None:
    """
    Process all video pairs for a single cubicle.

    Args:
        args: Tuple containing (cubicle_name, videos, prompt_template, api_key)
    """
    cubicle_name, videos, prompt_template, api_key = args

    # Create pairs of consecutive videos
    video_pairs = create_video_pairs(videos)

    print(f"Started processing cubicle {cubicle_name} with {len(video_pairs)} video pairs")

    for (ep1_num, ep1_path), (ep2_num, ep2_path) in video_pairs:
        print(f"Processing {cubicle_name}: episodes {ep1_num} and {ep2_num}...")

        # Prepare the prompt
        prompt = prepare_prompt(prompt_template, cubicle_name, ep1_num, ep2_num)

        # Generate response from Gemini
        response = gemini_response(api_key, prompt, ep1_path, ep2_path)

        # Extract JSON content
        json_content = extract_json_from_response(response)

        # Save the JSON response
        output_filename = f"{cubicle_name}_{ep1_num}_{ep2_num}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        save_json_response(json_content, output_path)

        print(f"Saved response to {output_path}")

    print(f"Completed processing cubicle {cubicle_name}")


def process_video_pairs(cubicle_videos: Dict[str, List[Tuple[int, str]]], prompt_template: str, api_key: str) -> None:
    """
    Process all video pairs for each cubicle in parallel.

    Args:
        cubicle_videos: Dict mapping cubicle names to video information
        prompt_template: The prompt template string
        api_key: The Gemini API key
    """
    # Prepare arguments for parallel processing
    process_args = [(cubicle_name, videos, prompt_template, api_key) for cubicle_name, videos in cubicle_videos.items()]

    # Determine the number of processes to use
    num_processes = min(MAX_PROC, len(cubicle_videos))
    print(f"Processing {len(cubicle_videos)} cubicles using {num_processes} processes")

    # Create a multiprocessing pool and process cubicles in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_single_cubicle, process_args)

    print("All cubicles have been processed")


def gemini_output():
    # Load API key
    api_key = os.getenv("GEMINI_API_KEY")

    # Load prompt template
    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get cubicle video pairs
    cubicle_videos = get_cubicle_video_pairs(VIDEOS_ROOT_DIR)

    if not cubicle_videos:
        print("No matching videos found.")
        return

    print(f"Found {len(cubicle_videos)} cubicles with videos.")

    # Process video pairs
    process_video_pairs(cubicle_videos, prompt_template, api_key)

    print("Local Temporal Evaluation completed successfully!")


def read_csv_file(file_path: str) -> List[Dict[str, str]]:
    """Read a CSV file and return its content as a list of dictionaries."""
    import csv

    data = []
    try:
        with open(file_path, "r") as f:
            # Use CSV reader to handle comma-separated values properly
            reader = csv.reader(f)
            lines = list(reader)
            if not lines:
                return data

            header = [col.strip() for col in lines[0]]

            for line in lines[1:]:
                if len(line) == len(header):
                    row_data = {}
                    for i, col in enumerate(header):
                        row_data[col] = line[i].strip()
                    data.append(row_data)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return data


def gt_gen():
    """Generate ground truth JSON files from CSV files."""
    print("Starting ground truth generation...")

    # Set output directory for ground truth files
    ground_truth_dir = os.path.join(OUTPUT_DIR, "ground_truth")
    os.makedirs(ground_truth_dir, exist_ok=True)

    # Walk through all cubicle folders in CSV_ROOT_DIR
    cubicle_folders = [folder for folder in os.listdir(CSV_ROOT_DIR) if os.path.isdir(os.path.join(CSV_ROOT_DIR, folder))]

    for cubicle_name in cubicle_folders:
        print(f"Processing cubicle: {cubicle_name}")
        cubicle_path = os.path.join(CSV_ROOT_DIR, cubicle_name)

        # Read all CSV files
        object_counting_csv = os.path.join(cubicle_path, "Object Counting.csv")
        object_detection_csv = os.path.join(cubicle_path, "Object Detection.csv")
        object_location_csv = os.path.join(cubicle_path, "Object Location Change.csv")
        object_state_csv = os.path.join(cubicle_path, "Object State Change.csv")

        # Load data from the CSV files
        counting_data = read_csv_file(object_counting_csv)
        detection_data = read_csv_file(object_detection_csv)
        location_data = read_csv_file(object_location_csv)
        state_data = read_csv_file(object_state_csv)

        # Organize data by episode
        changes_by_episode = {}

        # Process Object Counting data
        for item in counting_data:
            episode = int(item["Episode"])
            if episode > 0:
                if episode not in changes_by_episode:
                    changes_by_episode[episode] = []

                changes_by_episode[episode].append(
                    {
                        "Object": item["Non-unique Object"],
                        "Change Type": "Object Counting",
                        "Change Detail": f"count changed from {item['Initial Count']} to {item['Final Count']}",
                    }
                )

        # Process Object Detection data
        for item in detection_data:
            episode = int(item["Episode"])
            if episode > 0:
                if episode not in changes_by_episode:
                    changes_by_episode[episode] = []

                change_type = item["Appear/ Disappear (Change)"]
                if "Appear" in change_type:
                    detail = f"appeared at {item['Location inside Cubicle']}"
                elif "Disappear" in change_type:
                    detail = f"disappeared from {item['Location inside Cubicle']}"
                else:
                    raise ValueError(f"Unexpected change type: {change_type}")

                changes_by_episode[episode].append(
                    {"Object": item["Unique Object"], "Change Type": "Object Detection", "Change Detail": detail}
                )

        # Process Object Location Change data
        for item in location_data:
            episode = int(item["Episode"])
            if episode > 0:
                if episode not in changes_by_episode:
                    changes_by_episode[episode] = []

                changes_by_episode[episode].append(
                    {
                        "Object": item["Unique Object"],
                        "Change Type": "Object Location Change",
                        "Change Detail": f"moved from {item['Initial Location Inside Cubicle']} to {item['Final Location Inside Cubicle']}",
                    }
                )

        # Process Object State Change data
        for item in state_data:
            episode = int(item["Episode"])
            if episode > 0:
                if episode not in changes_by_episode:
                    changes_by_episode[episode] = []

                changes_by_episode[episode].append(
                    {
                        "Object": item["Object"],
                        "Change Type": "Object State Change",
                        "Change Detail": f"state changed from {item['Initial State']} to {item['Final State']}",
                    }
                )

        # Generate JSON files for each episode pair
        for episode in changes_by_episode:

            # Format the changes with C1, C2, etc. as keys
            formatted_changes = {}
            for i, change in enumerate(changes_by_episode[episode], 1):
                change_key = f"C{i}"
                formatted_changes[change_key] = change

            # Create the output filename
            output_filename = f"{cubicle_name}_{episode-1}_{episode}.json"
            output_path = os.path.join(ground_truth_dir, output_filename)

            # Save the JSON file
            with open(output_path, "w") as f:
                json.dump(formatted_changes, f, indent=2)

            print(f"Created ground truth file: {output_filename}")

    print("Ground truth generation completed successfully!")


def process_single_match(args: Tuple[str, str, str, str]) -> None:
    """
    Process a single match between ground truth and Gemini output JSON files.

    Args:
        args: Tuple containing (gt_file_path, gemini_output_dir, change_pair_dir, prompt_template, api_key)
    """
    gt_file_path, gemini_output_dir, change_pair_dir, prompt_template, api_key = args

    # Extract the filename
    filename = os.path.basename(gt_file_path)

    # Find the corresponding Gemini output file
    gemini_output_file_path = os.path.join(gemini_output_dir, filename)
    if not os.path.exists(gemini_output_file_path):
        print(f"Skipping {filename}: No matching Gemini output file found")
        return

    print(f"Processing {filename}...")

    # Load the ground truth JSON content
    with open(gt_file_path, "r") as gt_file:
        gt_content = gt_file.read()

    # Load the Gemini output JSON content
    with open(gemini_output_file_path, "r") as gemini_file:
        gemini_content = gemini_file.read()

    # Create a copy of the prompt template
    full_prompt = prompt_template

    # Inserting JSON contents
    full_prompt = full_prompt.replace("## Output JSON\n", f"## Output JSON\n```json\n{gemini_content}\n```\n\n")
    full_prompt = full_prompt.replace("## Ground Truth JSON\n", f"## Ground Truth JSON\n```json\n{gt_content}\n```\n\n")

    # Create Gemini client
    client = genai.Client(api_key=api_key)

    # Generate response from Gemini
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=1.0,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )
    response = response.text.strip()

    json_content = extract_json_from_response(response)

    # Save the matched changes to the change_pair directory
    output_path = os.path.join(change_pair_dir, filename)
    save_json_response(json_content, output_path)

    print(f"Saved matched changes to {output_path}")


def change_match():
    """
    Match changes between ground truth and Gemini output JSON files.

    This function:
    1. Finds all JSON files in the ground truth directory
    2. Locates corresponding JSON files in the Gemini output directory
    3. Loads the evaluation prompt template
    4. Uses multiprocessing to parallelize the matching process
    5. Sends the prompt to Gemini for change matching analysis
    6. Saves the matched changes to the change_pair directory
    """
    ground_truth_dir = os.path.join(OUTPUT_DIR, "ground_truth")
    gemini_output_dir = os.path.join(OUTPUT_DIR, "gemini_output")
    change_pair_dir = os.path.join(OUTPUT_DIR, "change_pair")

    # Load API key
    api_key = os.getenv("GEMINI_API_KEY")

    # Load evaluation prompt template
    prompt_template = load_prompt_template(EVA_PROMPT)

    # Find all ground truth JSON files
    gt_files = glob.glob(os.path.join(ground_truth_dir, "*.json"))

    if not gt_files:
        print("No ground truth files found.")
        return

    print(f"Found {len(gt_files)} ground truth files to process.")

    # Prepare arguments for parallel processing
    process_args = [(gt_file, gemini_output_dir, change_pair_dir, prompt_template, api_key) for gt_file in gt_files]

    # Determine the number of processes to use
    num_processes = min(MAX_PROC, len(gt_files))
    print(f"Processing {len(gt_files)} files using {num_processes} processes")

    # Create a multiprocessing pool and process files in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_single_match, process_args)

    print("Change matching completed successfully!")


def change_match_check():
    """
    Checking whether each change pair json file is combined with correct change event in the
    corresponding ground truth and gemini output json files.
    """
    ground_truth_dir = os.path.join(OUTPUT_DIR, "ground_truth")
    gemini_output_dir = os.path.join(OUTPUT_DIR, "gemini_output")
    change_pair_dir = os.path.join(OUTPUT_DIR, "change_pair")

    def change_event_eq(event1: Dict[str, str], event2: Dict[str, str]) -> bool:
        """
        Check if two change events are equal based on their attributes.
        """
        return (
            unify_text(event1["Object"]) == unify_text(event2["Object"])
            and unify_text(event1["Change Type"]) == unify_text(event2["Change Type"])
            and unify_text(event1["Change Detail"]) == unify_text(event2["Change Detail"])
        )

    def found_in_json(json_data, change_event):
        for key, value in json_data.items():
            if change_event_eq(value, change_event):
                return key
        return None

    # iterating through all the change pair json files
    for file_name in os.listdir(change_pair_dir):
        change_pair_path = os.path.join(change_pair_dir, file_name)
        with open(change_pair_path, "r") as f:
            change_pair_data = json.load(f)

        ground_truth_path = os.path.join(ground_truth_dir, file_name)
        with open(ground_truth_path, "r") as f:
            gt_data = json.load(f)

        gemini_output_path = os.path.join(gemini_output_dir, file_name)
        with open(gemini_output_path, "r") as f:
            gemini_output_data = json.load(f)

        if "Matched Change" in change_pair_data:
            for change_key, match_change in change_pair_data["Matched Change"].items():
                # Verify Output in matched changes
                output_change = match_change["Output"]
                event_id = found_in_json(gemini_output_data, output_change)

                if event_id is not None:
                    del gemini_output_data[event_id]
                else:
                    print(f"Error in {file_name}: Output match not found for {change_key}:")
                    print(f"  Claimed match: {output_change}")

                # Verify Ground Truth in matched changes
                gt_change = match_change["Ground Truth"]
                event_id = found_in_json(gt_data, gt_change)

                if event_id is not None:
                    del gt_data[event_id]
                else:
                    print(f"Error in {file_name}: Ground Truth match not found for {change_key}:")
                    print(f"  Claimed match: {gt_change}")

        # Verify Output-only changes
        if "Only in Output" in change_pair_data:
            for change_key, output_change in change_pair_data["Only in Output"].items():
                event_id = found_in_json(gemini_output_data, output_change)

                if event_id is not None:
                    del gemini_output_data[event_id]
                else:
                    print(f"Error in {file_name}: Output-only entry not found for {change_key}:")
                    print(f"  Claimed entry: {output_change}")

        # Verify Ground Truth-only changes
        if "Only in Ground Truth" in change_pair_data:
            for change_key, gt_change in change_pair_data["Only in Ground Truth"].items():
                event_id = found_in_json(gt_data, gt_change)
                if event_id is not None:
                    del gt_data[event_id]
                else:
                    print(f"Error in {file_name}: Ground Truth-only entry not found for {change_key}:")
                    print(f"  Claimed entry: {gt_change}")

        # Check if there are any unmatched changes in the output and ground truth
        if gemini_output_data:
            print(f"Error in {file_name}: Unmatched changes in Output:")
            for change_key, change_event in gemini_output_data.items():
                print(f"  Unmatched entry: {change_key}: {change_event}")
        if gt_data:
            print(f"Error in {file_name}: Unmatched changes in Ground Truth:")
            for change_key, change_event in gt_data.items():
                print(f"  Unmatched entry: {change_key}: {change_event}")

        change_pair_data["Statistics"] = {
            "Number of Matched Change": len(change_pair_data.get("Matched Change", {})),
            "Number of Only in Output": len(change_pair_data.get("Only in Output", {})),
            "Number of Only in Ground Truth": len(change_pair_data.get("Only in Ground Truth", {})),
        }

        # Save the updated change pair data
        with open(change_pair_path, "w") as f:
            json.dump(change_pair_data, f, indent=2)


def f1_score(precision: float, recall: float) -> float:
    """
    Calculate the F1 score based on precision and recall.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def f1_calc():
    # iterate through all the change pair json files
    change_pair_dir = os.path.join(OUTPUT_DIR, "change_pair")

    # read an accumulate the statistics
    total_matched = 0
    total_gt_only = 0
    total_output_only = 0

    for file_name in os.listdir(change_pair_dir):
        change_pair_path = os.path.join(change_pair_dir, file_name)
        with open(change_pair_path, "r") as f:
            change_pair_data = json.load(f)

        # Accumulate statistics
        total_matched += change_pair_data["Statistics"]["Number of Matched Change"]
        total_gt_only += change_pair_data["Statistics"]["Number of Only in Ground Truth"]
        total_output_only += change_pair_data["Statistics"]["Number of Only in Output"]

    # Calculate precision and recall
    precision = total_matched / (total_matched + total_output_only) if total_matched + total_output_only > 0 else 0
    recall = total_matched / (total_matched + total_gt_only) if total_matched + total_gt_only > 0 else 0
    f1 = f1_score(precision, recall)
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    print(f"Total Matched Changes: {total_matched}")
    print(f"Total Only in Ground Truth: {total_gt_only}")
    print(f"Total Only in Output: {total_output_only}")


if __name__ == "__main__":

    # Get output from Gemini
    gemini_output()

    # Generate ground truth files
    gt_gen()

    # Match the change events between the ground truth and Gemini output json files
    change_match()

    # Verify the matched changes and add statistics
    change_match_check()

    # Calculate F1 score
    f1_calc()
