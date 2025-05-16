import os
import time
import json
from google import genai
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Determine which model to use
# print("List of models that support generateContent:\n")
# for m in client.models.list():
#     for action in m.supported_actions:
#         if action == "generateContent":
#             print(m.name)

# GEMIMI_MODEL = "gemini-2.0-flash"
GEMIMI_MODEL = "gemini-2.5-pro-preview-05-06"

# Classes to create a structure output from Gemini
class Cubicle(BaseModel):
    id: str
    name: str

class CubicleListResponse(BaseModel):
    count: int
    cubicles: list[Cubicle]

def save_to_json(episode_id, data):
    filename = f"episode_{episode_id}_output.json"
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Saved result for episode {episode_id} to {filename}")

for i in range(0, 6):  # From episode_0 to episode_5
    episode_path = f"global_changes_videos/episode_{i}_720p_10fps.mp4"
    print(f"Uploading and analyzing {episode_path}...")

    myfile = client.files.upload(file=episode_path)
    print(f"{myfile=}")

    # Wait until the file is processed
    while not myfile.state or myfile.state.name != "ACTIVE":
        print("Processing video...")
        print("File state:", myfile.state)
        time.sleep(5)
        myfile = client.files.get(name=myfile.name)

    # Prompt for the episode
    result = client.models.generate_content(
        model=GEMIMI_MODEL,
        contents=[
            myfile,
            "Provide me with a count of the number of cubicles seen in the video and list the cubicles with their id and name."
        ],
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": CubicleListResponse,
        },
    )
    # print(f"Episode {i} result: {result.text}")
    # print("=" * 60)

    # Parse the result into the structured response model
    response_data = CubicleListResponse.parse_raw(result.text)

    # Prepare the data to save
    result_data = {
        "count": response_data.count,
        "cubicles": [{"id": cubicle.id, "name": cubicle.name} for cubicle in response_data.cubicles],
    }

    # Save the result to a JSON file
    save_to_json(i, result_data)

    print(f"Episode {i} result: {response_data.count} cubicles found")
    for cubicle in response_data.cubicles:
        print(f"Cubicle ID: {cubicle.id}, Name: {cubicle.name}")
    
    print("=" * 60)
