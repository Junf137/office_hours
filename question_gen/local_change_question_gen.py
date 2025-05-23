from google import genai
from google.genai import types
import os
import re


def gemini_response(api_key: str, full_prompt: str) -> str:
    """
    Generate a response from Gemini model using the provided API key and prompt.
    This function initializes a Gemini client, creates content with the given prompt,
    and generates a response using the gemini-2.5-flash-preview model with specific
    configuration parameters.
    Args:
        api_key (str): The authentication API key for accessing the Gemini API.
        full_prompt (str): The text prompt to send to the model.
    Returns:
        str: The generated text response from the Gemini model, with whitespace trimmed.
    Note:
        This function uses the gemini-2.5-flash-preview-04-17 model with a temperature
        of 1.0 and includes thinking configuration.
    """

    client = genai.Client(api_key=api_key)

    # Create content parts
    text_content = types.Content(
        parts=[
            types.Part(text=full_prompt),
        ],
        role="user",
    )

    # Generate content with proper parameters
    response = client.models.generate_content(
        model="models/gemini-2.5-pro-preview-05-06",
        contents=text_content,
        config=types.GenerateContentConfig(
            temperature=1.0,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )

    return response.text.strip()


def json_parse(response_text: str, output_json_path: str, output_other_path: str) -> None:
    """
    Extract JSON content from a response text and save it to separate files.
    This function searches for content between ```json and ``` markers in the response text.
    It saves the JSON content to one file and all other content to another file.
    Args:
        response_text (str): The text containing potential JSON content enclosed in markdown code blocks.
        output_json_path (str): File path where the extracted JSON content will be saved.
        output_other_path (str): File path where the non-JSON content will be saved.
    Returns:
        None
    Note:
        - If no JSON content is found between the markdown markers, an empty JSON object is written to the JSON file.
        - The non-JSON content is saved only if it's not empty.
    """

    # Extract JSON content between ```json and ``` markers
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
    if json_match:
        json_content = json_match.group(1).strip() + "\n"
        # Replace the JSON block with empty string to get everything else
        other_content = re.sub(r"```json\s*[\s\S]*?\s*```", "", response_text).strip()
    else:
        json_content = "{}\n"
        other_content = response_text

    with open(output_json_path, "w") as json_file:
        json_file.write(json_content)

    if other_content:
        with open(output_other_path, "w") as other_file:
            other_file.write(other_content + "\n")


def process_csv_file(csv_path, api_key):
    """
    Process a single CSV file by finding the corresponding prompt file and generating
    response files in the same directory as the CSV file.

    Args:
        csv_path (str): Path to the CSV file to process
        api_key (str): Gemini API key

    Returns:
        None
    """
    # Extract the base name without extension (e.g., "Object Location Change")
    csv_filename = os.path.basename(csv_path)
    base_name = os.path.splitext(csv_filename)[0]

    # Find corresponding prompt file
    prompt_path = os.path.join("./prompt", f"{base_name}.md")

    # Set output paths in the same folder as the CSV file
    csv_dir = os.path.dirname(csv_path)
    output_json_path = os.path.join(csv_dir, f"{base_name}.json")
    output_other_path = os.path.join(csv_dir, f"{base_name} Other Response.txt")

    if not os.path.exists(prompt_path):
        print(f"Warning: Prompt file not found for {csv_path}. Skipping.")
        return

    print(f"Processing {csv_path}")
    print(f"  Using prompt: {prompt_path}")
    print(f"  Output JSON: {output_json_path}")
    print(f"  Output Other: {output_other_path}")

    # Build the full prompt
    full_prompt = ""
    with open(prompt_path, "r") as prompt_file:
        full_prompt += prompt_file.read().strip()
    with open(csv_path, "r") as csv_file:
        csv_content = csv_file.read()
        full_prompt += f"\n\n<<<{csv_filename} start\n{csv_content}\n{csv_filename} end>>>"

    # Generate and parse response
    response_text = gemini_response(api_key, full_prompt)
    json_parse(response_text, output_json_path, output_other_path)


def process_all_csv_files(root_folder, api_key):
    """
    Walk through a root folder and process all CSV files found.

    Args:
        root_folder (str): Root folder to start searching from
        api_key (str): Gemini API key

    Returns:
        int: Number of files processed
    """
    count = 0
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                csv_path = os.path.join(dirpath, filename)
                process_csv_file(csv_path, api_key)
                count += 1

    return count


if __name__ == "__main__":
    with open("./data/gemini_api_key.txt", "r") as f:
        api_key = f.read().strip()

    # Set root folder to process (in this case, the csv_files folder)
    root_folder = "./csv_files"

    # Process all CSV files
    files_processed = process_all_csv_files(root_folder, api_key)
    print(f"Processed {files_processed} CSV files.")
