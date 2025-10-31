import json
from pathlib import Path

orig_file = Path("../ground_truth/episode_0_gt.json")

for i in range(24):  # 0..23
    new_file = Path(f"episode_0_gt_part_{i}.json")
    
    # Read the original file as text
    text = orig_file.read_text()
    
    # Parse JSON
    data = json.loads(text)
    
    # Update count
    data["count"] = i + 1
    
    # Replace the "count" value in the original text to preserve spacing
    # This assumes "count" appears once in the file
    import re
    new_text = re.sub(r'"count"\s*:\s*\d+', f'"count": {i+1}', text)
    
    # Write the new file
    new_file.write_text(new_text)
