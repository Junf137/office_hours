from google import genai
from google.genai import types
import os
import json
import re
import time

# Fetch the API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Paths
image_folder_path = ""  # Replace with keyframe folder path
label_data_path = "" # Replace with semantic map path

# Load label data
with open(label_data_path, 'r') as f:
    label_data = json.load(f)

# Categories
CATEGORIES = [
    "Object detection / classification",
    "Object counting",
    "Object state changes (attribute)",
    "Object location",
    "Cubicle/room location"
]

# Get image files
image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith('.png')])

# Output containers
all_questions = {}
category_outputs = {cat: {} for cat in CATEGORIES}

# Context builder
def get_context_prefix(image_id):
    frame_data = label_data.get(image_id, {})
    robot_locations = frame_data.get("robot_location", [])
    viewed = [str(v.get("alias", v["label"])) for v in frame_data.get("viewed_locations", [])]
    landmarks = [str(lm) for lm in frame_data.get("landmarks", [])]

    context_description = []

    if robot_locations:
        robot_loc_text = ", ".join(robot_locations[:-1]) + f", and {robot_locations[-1]}" if len(robot_locations) > 1 else robot_locations[0]
        context_description.append(f"the robot is at {robot_loc_text}")

    if viewed:
        context_description.append(f"it is viewing: {', '.join(viewed)}")
    if landmarks:
        context_description.append(f"visible landmarks include: {', '.join(landmarks)}")

    return "Context: " + "; ".join(context_description) + "." if context_description else ""

# Main loop
for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    image_id = os.path.splitext(image_file)[0]

    prefix = get_context_prefix(image_id)

    prompt = f"""
    You are a highly accurate visual reasoning assistant.

    Your task is to generate **exactly one question per category** based strictly on the visible content of the image and the provided context. **Use natural and casual language that sounds like something a person would ask when viewing the image**.

    The categories are:
    - Object detection / classification
    - Object counting
    - Object state changes (attribute)
    - Object location
    - Cubicle/room location

    Context for the image:
    {prefix}

    Strict instructions:
    - **Only use objects, attributes, and relationships that are clearly visible in the image.** Do not guess or hallucinate.
    - Use the context (robot location, viewed areas, landmarks) to make the question specific. Viewed areas are based on the robot's perspective (closest to furthest cubicle).
    - Refer to cubicles using aliases when available (e.g., "Harish's cubicle"). Don't mention the robot's position in the question.
    - Avoid yes/no questions.
    - Questions must be focused on objects **inside cubicles**.
    - For "Cubicle/room location", use spatial comparisons (e.g., relative to another cubicle or a door).
    - Make sure the correct answer is visibly justifiable — otherwise skip that object/question.
    - Randomize the correct answer position (A–D). "E" must always be "None of the above".
    - **Do not include questions for any other categories. Return only the five specified.**

    Respond with five clearly separated JSON blocks, each prefixed by a header like this:

    [Object detection / classification]
    {{
    "Type": "Object detection / classification",
    "Image": "{image_id}",
    "Question": "...",
    "Multiple Choice": {{
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "...",
        "E": "None of the above"
    }},
    "Correct Choice": "<A|B|C|D|E>"
    }}

    Only use the categories listed. Do not invent new ones or return fewer/more than five total questions.
    """.strip()

    # Read image and send it using types.Part
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            top_p=1.0,
            response_modalities=["TEXT"]
        )
    )

    print(f"Response for {image_file}: {response.text}")
    time.sleep(1)

    # Parse response
    output = {cat: {} for cat in CATEGORIES}
    current_category = None
    question_block = ""

    for line in response.text.strip().splitlines():
        category_match = re.match(r"\[(.*?)\]", line.strip())
        if category_match:
            category_name = category_match.group(1)
            if category_name in CATEGORIES:
                if current_category and question_block.strip():
                    try:
                        parsed_json = json.loads(question_block.strip())
                        output[current_category] = parsed_json
                        category_outputs[current_category][image_file] = parsed_json
                    except json.JSONDecodeError:
                        print(f"JSON parse error for {image_file} under {current_category}")
                current_category = category_name
                question_block = ""
        else:
            question_block += line + "\n"

    # Final block
    if current_category and question_block.strip():
        try:
            parsed_json = json.loads(question_block.strip())
            output[current_category] = parsed_json
            category_outputs[current_category][image_file] = parsed_json
        except json.JSONDecodeError:
            print(f"Final JSON parse error for {image_file} under {current_category}")

    all_questions[image_file] = output

# Save outputs
for category, data in category_outputs.items():
    filename = category.lower().replace(" ", "_").replace("/", "_") + ".json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

print("All questions saved with context and split by category.")
