You are provided with two JSON files:
- Output JSON: Contains changes detected by a VLM between two video episodes.
- Ground Truth JSON: Contains human-labeled true changes between the same two video episodes.
Your task is to compare these two JSON files and identify matched and unmatched change events according to the following structured criteria:

## Step-by-step Instructions:
1. Matching Process:
   - For each change event in the Ground Truth, find if there's a corresponding event in the Output JSON by checking semantic similarity of the "Object" descriptions.
   - Consider two object descriptions as a match if they refer semantically to the same or very similar objects despite slight naming differences (e.g., "pair of shoes" and "pair of sneakers" should be matched, "Maroon eyeglasses case" and "glasses box" should be matched).
   - Each matched pair (one from Ground Truth, one from Output) should be stored clearly as a single entry under `"Matched Change"`.
2. Categorizing Unmatched Events:
   - If a Ground Truth event has no suitable match in Output, place it under `"Only in Ground Truth"`.
   - Similarly, Output events with no suitable Ground Truth counterpart should be listed under `"Only in Output"`.
3. Output JSON Structure:
   Provide your final evaluation strictly following this JSON structure (Increment IDs sequentially (C1, C2, etc.) separately within each of the three categories):

```json
{
    "Matched Change": {
        "C1": {
            "Output": {
                "Object": "",
                "Change Type": "",
                "Change Detail": ""
            },
            "Ground Truth": {
                "Object": "",
                "Change Type": "",
                "Change Detail": ""
            }
        }
    },
    "Only in Output": {
        "C1": {
            "Object": "",
            "Change Type": "",
            "Change Detail": ""
        }
    },
    "Only in Ground Truth": {
        "C1": {
            "Object": "",
            "Change Type": "",
            "Change Detail": ""
        }
    }
}
```
4. Additional Guidelines for Matching:
   - Matching is based primarily on object identity, taking into account synonyms and paraphrases.
   - The "Change Type" does NOT need to match to qualify as an initial object-level match (this will be checked in a separate detailed evaluation step later).
   - Do NOT attempt to match purely based on "Change Detail"; this will also be evaluated separately.
5. Constraints:
   - Output only valid JSON, no explanatory or additional text.
   - Be comprehensive: ensure every event from both files is included exactly once in the final result.

## Output JSON


## Ground Truth JSON


---
Proceed carefully and systematically, ensuring accuracy in matching and categorization according to semantic similarity of object descriptions.
