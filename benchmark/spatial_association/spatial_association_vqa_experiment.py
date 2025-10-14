import os
import time
import json
from typing import List
from google import genai
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

#  --- Gemini client setup ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Determine which model to use
# print("List of models that support generateContent:\n")
# for m in client.models.list():
#     for action in m.supported_actions:
#         if action == "generateContent":
#             print(m.name)

# GEMIMI_MODEL = "gemini-2.5-flash-lite"
# GEMIMI_MODEL = "gemini-2.5-flash"
GEMIMI_MODEL = "gemini-2.5-pro"

# --- Constants ---
NUM_EPISODES = 1

# --- Allowed relations ---
ALLOWED_REL = {"left_of", "right_of"}

# --- Pydantic schemas to structure VLM output ---
class CubiclePair(BaseModel):
    id: str
    name: str

class SimpleEdge(BaseModel):
    source: str   # owner name string (e.g., "Amy")
    target: str   # owner name string (e.g., "Jason")
    relation: str # "left_of" or "right_of"

class CombinedResponse(BaseModel):
    count: int                # number of cubicles with readable names
    cubicles: List[CubiclePair]  # list of id<->name pairs (only include cubicles that have readable names)
    edges: List[SimpleEdge]      # left_of / right_of relations between names

# --- Prompt ---
# TODO: keep this or nah: "count" must equal the number of entries in "cubicles".
# TODO: add some hinting for the directtions
# TODO: adjacancy
# TODO: left right and across 
PROMPT = """
You are given a video surveying an office with multiple cubicles. Produce ONE strict JSON object (no commentary) with exactly three keys: "count", "cubicles", and "edges".

Schema:
{
  "count": <integer>,
  "cubicles": [ {"id":"2008M","name":"Amy"}, ... ],
  "edges": [ {"source":"Amy","target":"Jason","relation":"left_of"}, ... ]
}

Rules:
- "cubicles" must include ONLY cubicles for which a readable owner name appears in the video. Each entry must contain an id (visible id like 2008 M) and the exact visible name string.
- "count" must equal the number of entries in "cubicles".
- "edges" must reference owner names (strings) and use ONLY the relations "left_of" or "right_of".
"""

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved", path)

if __name__ == "__main__":
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(NUM_EPISODES):
        episode_path = f"data/global_changes_videos/episode_{i}_720p_10fps.mp4"
        if not os.path.exists(episode_path):
            print("Missing file:", episode_path)
            continue

        print("Uploading", episode_path)
        myfile = client.files.upload(file=episode_path)

        # Wait until the file is processed
        while not myfile.state or myfile.state.name != "ACTIVE":
            print("Waiting for file processing... state:", myfile.state)
            time.sleep(3)
            myfile = client.files.get(name=myfile.name)

        # end request prompt for the episode
        resp = client.models.generate_content(
            model=GEMIMI_MODEL,
                contents=[myfile, PROMPT],
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_schema": CombinedResponse,
            },
        )

        # save raw response
        raw_path = os.path.join(out_dir, f"episode_{i}_raw.txt")
        with open(raw_path, "w") as f:
            f.write(resp.text)
        print("Saved raw response to", raw_path)

        # validate & parse
        try:
            parsed = CombinedResponse.model_validate_json(resp.text)
        except Exception as e:
            print("Validation failed:", e)
            print("Raw response:\n", resp.text)
            continue

        # basic checks
        if parsed.count != len(parsed.cubicles):
            print(f"Warning: count ({parsed.count}) != number of cubicles returned ({len(parsed.cubicles)}).")

        bad_rels = [e for e in parsed.edges if e.relation not in ALLOWED_REL]
        if bad_rels:
            print("Warning: unsupported relations found:", {e.relation for e in bad_rels})

        # check edge references exist in cubicle names
        cubicle_names = {c.name for c in parsed.cubicles}
        missing_refs = [e for e in parsed.edges if e.source not in cubicle_names or e.target not in cubicle_names]
        if missing_refs:
            print("Warning: some edges reference names not present in cubicles list:")
            for e in missing_refs:
                print(f"  {e.source} -> {e.target} ({e.relation})")

        # save parsed result
        out = {
            "count": parsed.count,
            "cubicles": [c.model_dump() for c in parsed.cubicles],
            "edges": [e.model_dump() for e in parsed.edges],
        }
        save_path = os.path.join(out_dir, f"episode_{i}_combined.json")
        save_json(save_path, out)

        # concise summary
        print(f"Episode {i}: named cubicles = {parsed.count}; edges = {len(parsed.edges)}")
        for c in parsed.cubicles:
            print(f"  {c.id} : {c.name}")
        for e in parsed.edges:
            print(f"  {e.source} -> {e.target} : {e.relation}")
        print("-" * 60)