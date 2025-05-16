You have been provided one CSV file named "Object Counting.csv", from a total of four possible CSV files:

1. Object Location Change.csv
2. Object Detection.csv
3. Object State Change.csv
4. Object Counting.csv

## Purpose and Context

The objective is to generate natural and realistic questions aimed at benchmarking VLM abilities to understand and identify object associations, appearances, and disappearances within realistic office cubicle scenes between consecutive video episodes.

## Explanation of CSV Files

- Each CSV file represents changes observed between consecutive videos.
  - Object Location Change.csv: Tracks movements of objects between locations within the cubicle.
  - Object Detection.csv: Tracks the appearance or disappearance of unique objects within the cubicle.
  - Object State Change.csv: Captures state transitions (initial to final state) of objects.
  - Object Counting.csv: Documents quantity changes of non-unique objects within the cubicle.
- Changes accumulate progressively across episodes (video n-1 to video n).
- Videos referenced across CSV files for the same episode are identical.

## Explanation of Columns in "Object Counting.csv"

- Episode: Indicates the videos between which changes occur. Episode n is from video (n-1) to video n.
- Non-unique Object: Specifies the object whose count has changed.
- Initial Count: Initial quantity of the specified object.
- Final Count: Final quantity of the specified object.

## Requirements for Generating Questions

Generate a JSON file containing questions based on every single recorded change within the provided CSV file. The JSON structure for each question must be as follows:

```json
{
    "Q1": {
        "Type": "Object Counting",
        "Change Number": 1,
        "Initial Video": 0,
        "Final Video": 1,
        "Question": "The question text",
        "Multiple Choice": {
            "A": "Possible Answer A",
            "B": "Possible Answer B",
            "C": "Possible Answer C",
            "D": "Possible Answer D",
            "E": "None of the above"
        },
        "Correct Choice": "A"
    }
}
```

## Detailed Instructions

1. Increment each question entry label sequentially (Q1, Q2, etc.).

2. Clearly specify "Type" from: Object Location Change, Object Detection, Object Counting, Object State Change. For the current CSV provided ("Object Counting.csv"), type is always "Object Counting".

3. "Change Number": Clearly corresponds to each row entry in the CSV for easy reference.

4. "Initial Video" and "Final Video": Always denote consecutive episodes (initial video i and final video i+1).

5. Provide exactly five multiple-choice options (A-E). Option E should be reserved for responses like "None of the above" or "All of the above," and should be correctly used occasionally to ensure difficulty.

6. False Answers (A-D):
   - Use only objects explicitly mentioned within the provided CSV file.
   - For other CSV types ("Object Location Change", "Object Detection", "Object State Change"), only use unique objects stated within their respective CSV files.

7. Do not explicitly state the video numbers within the question text itself.

8. If multiple changes occur within the same episode interval (e.g. 2 or 3 objects changed location), create multiple distinct questions ensuring each has only one correct answer.

9. Language and tone of the questions should feel conversational, natural, and realistic—similar to how a human would genuinely query a VLM about scene changes. Avoid robotic or unnatural phrasing.

10. Difficulty Levels:
    - Level 1 (hardest): Do not mention the specific object or the change explicitly. Example:
      > "Did anything change in the cubicle between these episodes?"
    - Level 2 (medium): Specify the object explicitly, ask about the type of change. Example:
      > "There was a change in the number of steel mugs in the cubicle. What exactly changed?"
      > "The mouse in the cubicle has been touched, what happened to it?"
    - Level 3 (easiest): Clearly state the type of change and ask about the specific object. Example:
      > "An object's quantity increased between the episodes. Which object increased in number?"
    Ensure a balanced mix of difficulty levels unless otherwise instructed.

11. For "Object Counting" questions specifically:
    - Avoid unnatural phrasing like "Which item's count changed from 1 to 2?".
    - Use natural phrasing focusing on the overall quantity change or scenario rather than overly numeric specifics.

## Examples:
1. Object Counting (Medium):
   > Question: "How did the quantity of steel mugs change?"
   > Multiple Choice:
   > - A: Increased from 1 to 2
   > - B: Decreased from 2 to 1
   > - C: Increased from 0 to 2
   > - D: Decreased from 3 to 1
   > - E: None of the above

2. Object Counting (Hard):
   > Question: "Did anything change in terms of the number of objects between these episodes? If so, what changed?"
   > Multiple Choice:
   > - A: The number of plastic bottles increased.
   > - B: The number of steel mugs decreased.
   > - C: The number of keyboards stayed the same.
   > - D: The number of chairs increased.
   > - E: None of the above

3. Object Location Change (Easy) -(for illustration only)-:
   > Question: "An object moved between episodes. Which object was it?"
   > Multiple Choice:
   > - A: Blue notebook
   > - B: Wireless keyboard
   > - C: Steel mug
   > - D: Chair
   > - E: Nothing moved
