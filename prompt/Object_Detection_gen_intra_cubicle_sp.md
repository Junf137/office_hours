
You are given the Object Detection CSV file, which is one of three CSV files that describe changes between consecutive videos in a multi‑video dataset:

| CSV file                   | What it describes                              | Example                                     |
| -------------------------- | ---------------------------------------------- | ------------------------------------------- |
| **Object Detection**       | Appearance or disappearance of a unique object | “New coffee machine appeared”               |
| **Object Counting**        | Change in the count of a non‑unique object     | “Number of blue pens decreased from 5 to 3” |
| **Object State Change**    | State transition of a unique object            | “Monitor changed from *off* to *on*”        |

The changes are cumulative meaning we are adding changes from video to video.
All the videos are the same between the CSV files. Meaning the videos representing episode 1 are the same across all csv files provided.


#### Your task

From the single CSV I’ve provided, create **one multiple‑choice question for *every* recorded change**.
These questions will be used to benchmark a Vision‑Language Model (VLM).

---

#### Question requirements
Note: word inside [] mean variables.
1. The question must follow one the following format:
    * Which object is in [cubicle]'s cubicle? 
    - Answer must include the [Unique object].
    * For both question and answer please make sure they have proper english grammar and sound normal, like how a human would talk.
2. Cover every change in the CSV once and only once.
5. No video numbers in the question text.
6. Choices: exactly five (A–E).
   * Use only objects listed in the CSV, make sure other object that appear on the fake answers are not in the same cubicle within the same episode.
   * Choice E must be a catch‑all such as “None of the above” or something similar. 
   * From time to time the correct answer must be E.
7. Correctness: specify the correct letter.
---



#### JSON Requirements

1. **Type** must be one of:
   * `Object Detection`
   * `Object Counting`
   * `Object State`
2. **Change** number must be present in the JSON output for every question
3.  **Video** number must be present in the JSON. The video number will be calculated by looking at the **Change appear or disappear** column. If **Change appear or disappear** = disappeared **Video** = **Episode** -1. If **Change appear or disappear** = appeared    **Video** = **Episode**. 
4. The question must be present on the JSON format. 
5. The multiple choice questions must be present. 
6. The correct choice must be present. 



#### Output format (JSON)

Please follow the following format. 

```json
{
  "Q1": {
    "Type": "Object Detection",
    "Change Number": 7,
    "Video": 3,
    "Question": "What object is in Amy's Cubicle?",
    "Multiple Choice": {
      "A": "A red panda plushy.",
      "B": "The dixit board game.",
      "C": "A lamp.",
      "D": "A toy of roman soldier head.",
      "E": "None of the above."
    },
    "Correct Choice": "B"
  },
  "...": {}
}
```
Please do not execute any code to do this. Just read the csv line by line and make the questions. 
Finally make sure the question has proper english grammar. 
*Repeat this structure for every change in the CSV.*
