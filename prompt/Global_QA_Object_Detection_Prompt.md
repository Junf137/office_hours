
You are given the Object Detection CSV file, which is one of four CSV files that describe changes between consecutive videos in a multi‑video dataset:

| CSV file                   | What it describes                              | Example                                     |
| -------------------------- | ---------------------------------------------- | ------------------------------------------- |
| **Object Location Change** | Where a unique object moved                    | “Black mug moved from desk A to desk B”     |
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
    * Which object has [appeared/disappeared] [in/from] [cubicle]'s cubicle? 
    - Answer must include the [location-inside-cubicle] and [object].
    * A/An [object]  has appeared in the office, In which cubicle did it appeared?
		- Answer must the [cubicle] and [location-inside-cubicle].	
    * A/An [object]  has disappeared from the office, From which cubicle did it disappeared?
    - Answer must the [cubicle] and [location-inside-cubicle].
    * For both question and answer please make sure they have proper english grammar and sound normal, like how a human would talk. Example: Questions "A Catan Board game box red has appeared in the office. In which cubicle did it appear?", Answer: "It appeared on Json's cubicle on the right side of the cubicle". Another example: Question: "An Eiffel tower toy has disappeared from the office. From which cubicle did it disappear?", Answer: "It disappeared from Jerry's cubicle it was at the right side of the cubicle. 
2. Cover every change in the CSV once and only once.
5. No video numbers in the question text.
6. Choices: exactly five (A–E).
   * Use only objects listed in the CSV.
   * Choice E must be a catch‑all such as “None of the above” or something similar. 
   * From time to time the correct answer must be E.
7. Correctness: specify the correct letter.
---



#### JSON Requirements

1. **Type** must be one of:
   * `Object Location Change`
   * `Object Detection`
   * `Object Counting`
   * `Object State Change`
2. Change number must be present in the JSON output for every question
3. Intial Video and Final video must be present on JSON output with the following format. Final_video = Episode and Initial Video = Episode - 1. Therefore if episode is 1 then intial video is 0 and final video is 1. 
4. The question must be present on the JSON format. 
5. The multiple choice questions must be present. 
6. The correct choice must be present. 



#### Output format (JSON)

Please follow the following format. 

```json
{
  "Q1": {
    "Type": "Object State Change",
    "Change Number": 7,
    "Initial Video": 3,
    "Final Video": 4,
    "Question": "What change occurred to the desk lamp?",
    "Multiple Choice": {
      "A": "It moved from the left shelf to the center shelf.",
      "B": "It changed from 'off' to 'on'.",
      "C": "A second lamp appeared next to it.",
      "D": "It was replaced by a white mug.",
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
