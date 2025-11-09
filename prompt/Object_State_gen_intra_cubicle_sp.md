
You are given the Object State Change CSV file, which is one of three CSV files that describe changes between consecutive videos in a multi‑video dataset:

| CSV file                   | What it describes                              | Example                                     |
| -------------------------- | ---------------------------------------------- | ------------------------------------------- |
| **Object Detection**       | Appearance or disappearance of a unique object | “New coffee machine appeared”               |
| **Object Counting**        | Change in the count of a non‑unique object     | “Number of blue pens decreased from 5 to 3” |
| **Object State Change**    | State transition of a unique object            | “Monitor changed from *off* to *on*”        |

The changes are cumulative meaning we are adding changes from video to video.
All the videos are the same between the CSV files. Meaning the videos representing episode 1 are the same across all csv files provided.


#### Your task

From the single CSV I’ve provided, create **two multiple‑choice question for *every* recorded change**.
These questions will be used to benchmark a Vision‑Language Model (VLM).

---

#### Question requirements
Note: word inside [] mean variables.
1. The question must follow the following format:
    * What is the state of the [Object] in [Cubicle]'s Cubicle? 
    * Make sure the fake answer are not true if the correct state is true. For example for a laptop let's say the correct state is a on monitor, then we can't put opened as a fake answer since to see the monitor the laptop must be opened thus both could be correct. 
    * For both question and answer please make sure they have proper english grammar and sound normal, like how a human would talk.
    * Make sure the answer to the question is not always A or B. The likelyhood of being A, B, C, D, or E should be equal. 
2. For each change change there will be two questions, one where the answer is the [Initial State] and another one where the answer is the [Final State]. We will differentiate between two question using the video key in the json. 
5. No video numbers in the question text.
6. Choices: exactly five (A–E).
   * For all the fake answers use a random number between 0 and 10.
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
3.  **Video** number must be present in the JSON. The **Video** = **Episode** - 1 if we are using **Initial State** as the answer & **Video** = **Episode** if we are using the **Final State** as the answer.  
4. The question must be present on the JSON format. 
5. The multiple choice questions must be present. 
6. The correct choice must be present. 



#### Output format (JSON)

Please follow the following format. 

```json
{
  "Q1": {
    "Type": "Object State",
    "Change Number": 7,
    "Video": 3,
    "Question": "What is the state of the laptop in Emily's",
    "Multiple Choice": {
      "A": "Monitor is on",
      "B": "Monitor is off",
      "C": "Closed Upside down",
      "D": "Closed upright",
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
