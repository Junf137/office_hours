# Office Hours: A Multiday Office Cubicle Dataset for Associative Embodied VQA

This repository contains a benchmark dataset and evaluation framework for video understanding tasks, focusing on both local and global temporal changes in videos.

## Overview

The benchmark dataset is designed to evaluate models' capabilities in understanding and reasoning about temporal changes in videos. It includes two main types of evaluations:

1. **Local Temporal Understanding**: Evaluates the model's ability to detect and understand specific changes within video segments.
2. **Global Video Understanding**: Assesses the model's comprehension of overall video content and temporal relationships.

## Directory Structure

```
.
├── benchmark/
│   ├── local_temporal_evaluation.py    # Evaluation script for local temporal changes
│   ├── video_understanding_global.py   # Global video understanding evaluation
│   ├── video_understanding_global_with_map.py  # Global evaluation with mapping
│   └── video_understanding_local.py    # Local video understanding evaluation
├── question_gen/
│   └── local_changes/                  # Question generation for local temporal changes
└── utils/                             # Utility functions and helpers
```

## Features

- **Local Temporal Evaluation**: Tests the model's ability to detect and understand specific changes within video segments
- **Global Video Understanding**: Evaluates overall video comprehension and temporal relationships
- **Question Generation**: Tools for generating evaluation questions for local temporal changes
- **Comprehensive Evaluation Metrics**: Multiple metrics to assess model performance

## Usage

### Local Temporal Evaluation

```python
python benchmark/local_temporal_evaluation.py
```

### Global Video Understanding

```python
python benchmark/video_understanding_global.py
```

### Global Video Understanding with Mapping

```python
python benchmark/video_understanding_global_with_map.py
```

## Requirements

- Python 3.x
- Required packages (to be listed in requirements.txt)

## Contributing

Contributions to improve the benchmark dataset or evaluation framework are welcome. Please feel free to submit issues and pull requests.

## License

[Specify your license here]

## Contact

For questions or suggestions, please contact Fernando J. Pena Cantu [fjpenaca@uwaterloo.ca] 