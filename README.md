# Office Hours: A Multiday Office Cubicle Dataset for Associative Embodied VQA
This repository contains a benchmark dataset and evaluation framework for video understanding tasks, focusing on both local and global temporal changes in videos.

## Overview
The benchmark dataset is designed to evaluate models' capabilities in understanding and reasoning about temporal changes in videos. It includes two main types of evaluations:

1. **Local Temporal Understanding**: Evaluates the model's ability to detect and understand specific changes within video segments.
2. **Global Video Understanding**: Assesses the model's comprehension of overall video content and temporal relationships.


## Dependencies
Creating conda environment and installing required dependencies.
```shell
conda create -n office_hours python=3.11
conda activate office_hours
pip install -r requirements.txt
```

## Directory Structure
```
.
├── benchmark/
│   ├── spatial_association/
│   │   ├── downsample_video.py                     # Downsample global videos to 720p resolution
│   │   └── spatial_association_vqa_experiment.py   # Evaluation script for global video spatial association experiment
│   ├── static_association_semantic_mapping/
│   │   ├── vid0_gemini_script.py                   # Generates visual reasoning questions from keyframes
│   │   └── vid0_label_map_to_frame.json            # JSON mapping of keyframes to robot location and viewed cubicles
│   ├── temporal_association/
│   │   └── local_temporal_evaluation.py            # Evaluation script for local temporal changes
│   ├── video_understanding_global.py               # Global video understanding evaluation
│   ├── video_understanding_global_with_map.py      # Global evaluation with mapping
│   └── video_understanding_local.py                # Local video understanding evaluation
├── question_gen/
│   └── local_change_question_gen.py                # Question generation for local temporal changes
├── prompt/                 # prompt used in benchmarking and question generation
├── keyframe_extraction/
|   ├── MASt3R-SLAM/        # A fork of MASt3R-SLAM to allow full resolution saving of keyframes
|   ├── split_videos.bash   # Script to split global videos into 5 minute segments with 30sec overlap
|   └── run_videos.bash     # Script to run MASt3R-SLAM on segmented videos
└── utils/                  # Utility functions and helpers
```

## Features
- **Local Temporal Evaluation**: Tests the model's ability to detect and understand specific changes within video segments
- **Global Video Understanding**: Evaluates overall video comprehension and temporal relationships
- **Question Generation**: Tools for generating evaluation questions for local temporal changes
- **Comprehensive Evaluation Metrics**: Multiple metrics to assess model performance

## Benchmarking

### Setting Up Gemini API Key
Before running scripts that requiring gemini API key, setting the api key in the environment variable.
```shell
export GEMINI_API_KEY=YOUR_API_KEY
```

### Spatial Association VQA Evaluation
```python
# Downsample videos
python benchmark/spatial_association/downsample_video.py

# evaluation
python benchmark/spatial_association/spatial_association_vqa_experiment.py
```

### Static Association-Semantic Mapping VQA Evaluation
Implemented a system for generating visual reasoning questions from video keyframes. This system:
```python
image_folder_path = "path/to/keyframes"
label_data_path = "path/to/vid0_label_map_to_frame.json"

python benchmark/static_association_semantic_mapping/vid0_gemini_script.py
```

### Temporal Association VQA Evaluation
```python
python benchmark/temporal_association/local_temporal_evaluation.py
```

### Single-Cubicle-Multi-Temporal VQA Evaluation
```python
python benchmark/video_understanding_local.py
```

### Multi-Cubicle-Multi-Temporal VQA Evaluation
```python
python benchmark/video_understanding_global.py
```

### Multi-Cubicle-Multi-Temporal VQA with Mapping Evaluation
```python
python benchmark/video_understanding_global_with_map.py
```

## Question Generation

### Local Change Videos Questions Generation
```python
python question_gen/local_change_question_gen.py
```

### Global Change Videos Questions Generation
Used prompt:
- Global_QA_Object_Count_Prompt.md
- Global_QA_Object_Detection_Prompt.md
- Global_QA_Object_Location_Change_Prompt.md
- Global_QA_Object_State_Prompt.md

## Contributing
Contributions to improve the benchmark dataset or evaluation framework are welcome. Please feel free to submit issues and pull requests.

## Contact
For questions or suggestions, please contact [Fernando J. Pena Cantu](mailto:fjpenaca@uwaterloo.ca).
