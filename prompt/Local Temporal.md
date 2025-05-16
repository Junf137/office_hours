You are provided with two consecutive video episodes ([episode_name_1] and [episode_name_2]) depicting an office scene. Your task is to closely observe cubicle [cubicle_name] in both episodes and list every single change that occurred inside this cubicle in the structured JSON format provided below.
```json
{
  "C1": {
    "Object": "object_name",
    "Change Type": "change_type",
    "Change Detail": "change_detail"
  },
}
```

### Requirements:
1. Change IDs ("C1", "C2", …): Sequential numbering for each observed change event.
2. "Object": Provide the exact name of the object affected by the change.
3. "Change Type": Clearly categorize each observed event into exactly one of the following types:
  - `Object Location Change`: An object has moved to a new location inside the cubicle.
  - `Object Detection`: A unique object has newly appeared or disappeared from the cubicle.
  - `Object State Change`: An object's condition or state has transitioned from one state to another.
  - `Object Counting`: The quantity of a non-unique object inside the cubicle has changed.
4. "Change Detail": Always follow the corresponding canonical format:
  - Object Location Change: `"moved from <initial_location> to <final_location>"`
  - Object Detection: `"appeared at <location>"` or `"disappeared from <location>"`
  - Object State Change: `"state changed from <initial_state> to <final_state>"`
  - Object Counting: `"count changed from <initial_count> to <final_count>"`

### Detailed Instructions:
1. Observe closely:
   - Start by identifying and listing all visible objects in cubicle [cubicle_name] during [episode_name_1].
   - For each object identified in the first episode, carefully track whether it moves, disappears, changes state, or changes count in [episode_name_2].
   - Also, check carefully for any newly appearing objects in [episode_name_2] that were not previously present in [episode_name_1].
2. Exclude irrelevant details:
   - Do NOT include changes involving chairs, as chairs frequently change by default and are not relevant.
   - ONLY document changes occurring inside cubicle [cubicle_name]. Ignore changes outside the specified cubicle entirely.
3. Handle viewpoint changes carefully:
   - Camera angles may vary significantly between the two episodes. Take extra care to recognize and document every possible observable change, even when viewing angles differ substantially.
4. Output constraints:
   - ONLY output valid JSON as specified.
   - Do NOT provide explanatory text, commentary, or reasoning outside the requested JSON structure.


Let's proceed carefully step-by-step, observing thoroughly and precisely documenting each relevant change.
