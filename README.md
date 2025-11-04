# CrazyCPM

**Critical Path Method and PERT Analysis Library**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)

A comprehensive Python library for project management analysis using Critical Path Method (CPM) and Program Evaluation and Review Technique (PERT). CrazyCPM provides network analysis capabilities with statistical modeling and professional visualization.

## License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Features

### Core Analysis
 * **Critical Path Method (CPM)** - Identify critical activities and project duration
 * **PERT Analysis** - Statistical modeling with uncertainty quantification
 * **Multiple Duration Formats** - Support for various input methods:
  * Direct duration and variance
  * Three-point PERT estimates (optimistic, most likely, pessimistic)
  * Two-point PERT estimates (optimistic, pessimistic)

### Advanced Capabilities
 * **Variance Propagation** - Statistical analysis of project completion times
 * **Probabilistic Estimates** - Quantile estimates and completion probabilities
 * **Network Optimization** - Automatic dummy activity generation
 * **Topological Sorting** - Efficient network traversal algorithms

### Technical Features
 * **High Performance** - C backend for computational efficiency
 * **Flexible Input Formats** - Multiple link dependency formats
 * **Comprehensive Export** - Dictionary and pandas DataFrame output
 * **Professional Visualization** - Graphviz-based network diagrams
 * **Error Handling** - Robust validation and error reporting

## Installation

### Prerequisites
 * Python 3.7 or higher
 * C++ compiler (GCC, Clang, or MSVC)
 * Graphviz system installation

### Install from GitHub

```bash
# Clone the repository
git clone https://github.com/shkolnick-kun/crazy_cpm.git
cd crazycpm
```

# Install system dependencies (Ubuntu/Debian)
```bash
sudo apt-get install graphviz build-essential cython
```

# Install
```bash
pip install .
```

Dependencies

The package requires the following Python dependencies:
 * numpy
 * pandas
 * graphviz
 * betapert
 * cython

# Quick Start
## Basic Usage
```python

from crazy_cpm import NetworkModel

# Define your Work Breakdown Structure (WBS)
wbs = {
    1: {'letter': 'A', 'duration': 5.0, 'variance': 1.0, 'name': 'Design'},
    2: {'letter': 'B', 'duration': 3.0, 'name': 'Development'},
    3: {'letter': 'C', 'optimistic': 2, 'most_likely': 3, 'pessimistic': 5, 'name': 'Testing'}
}

# Define activity dependencies
links = [[1, 2], [2, 3]]  # A → B → C

# Create and analyze the network model
model = NetworkModel(wbs, links=links)

# Get results as DataFrames
activities_df, events_df = model.to_dataframe()

# Generate visualization
model.viz('project_network')
```

## Advanced PERT Analysis
```python

# Mixed duration formats with probabilistic analysis
wbs_advanced = {
    1: {'letter': 'A', 'optimistic': 3, 'most_likely': 5, 'pessimistic': 8},
    2: {'letter': 'B', 'optimistic': 2, 'pessimistic': 6},
    3: {'letter': 'C', 'duration': 4.0, 'variance': 0.5}
}

# Create model with custom probability level
model_pert = NetworkModel(wbs_advanced, links=links, p=0.95)

# Access probabilistic estimates
activity = model_pert.get_activity_by_wbs_id(1)
print(f"95% quantile early start: {activity.early_start_pqe}")
```

## Input Formats
### Activity Duration Specifications

Three-point PERT:
```python

activity_data = {
    'letter': 'X',
    'optimistic': 3.0,      # Best-case scenario
    'most_likely': 5.0,     # Most probable duration
    'pessimistic': 8.0      # Worst-case scenario
}
```

Two-point PERT:
```python

activity_data = {
    'letter': 'Y',
    'optimistic': 2.0,
    'pessimistic': 6.0      # most_likely is calculated automatically
}
```

Direct Parameters:
```python

activity_data = {
    'letter': 'Z',
    'duration': 4.5,        # Mean duration
    'variance': 0.25        # Duration variance (optional)
}
```

### Dependency Link Formats

Two-row format:
```python

links = [
    [1, 2, 3],  # Source activities
    [2, 3, 4]   # Destination activities
]
```

Two-column format:
```python

links = [
    [1, 2],     # Activity 1 → Activity 2
    [2, 3],     # Activity 2 → Activity 3
    [3, 4]      # Activity 3 → Activity 4
]
```

Dictionary format:
```python

links = {
    'src': [1, 2, 3],
    'dst': [2, 3, 4]
}
```

# API Reference
## NetworkModel Class

The main class for network analysis:
```python

model = NetworkModel(
    wbs_dict,           # Work Breakdown Structure dictionary
    lnk_src=None,       # Source activities (old format)
    lnk_dst=None,       # Destination activities (old format)
    links=None,         # Dependency links (new formats)
    p=0.95,             # Probability level for quantile estimates
    default_risk=0.3,   # Default risk factor
    debug=False         # Enable debug mode
)
```

### Key Methods

 * `to_dataframe()` - Export results to pandas DataFrames
 * `to_dict()` - Export results to dictionary format
 * `viz(output_path)` - Generate network visualization


### Output Examples
#### Activities DataFrame
|  id  | wbs_id |letter |duration |variance | early_start | late_start  | reserve  |     name       |
| :--- | :----: | :---: | :-----: | :-----: | :---------: | :---------: | :------: | -------------: |
| 1    | 1      | A     | 5.0     | 1.0     | 0.0         | 0.0         | 0.0      | Design         |
| 2    | 2      | B     | 3.0     | 0.0     | 5.0         | 5.0         | 0.0	   | Development    |
#### Events DataFrame
|  id  |  stage  |  early  |  late   | reserve  |
| :--- | :-----: | :-----: | :-----: | -------: |
|1     | 0       | 0.0     | 0.0     | 0.0      |
|2     | 1       | 5.0     | 5.0     | 0.0      |
|3     | 2       | 8.0     | 8.0     | 0.0      |

### Visualization

The library generates professional network diagrams using Graphviz:
```python

# Generate and save visualization
dot = model.viz('my_project')

# Customize output path
model.viz(output_path='/path/to/project_network')
```

The visualization includes:
 * Red nodes/edges: Critical path (zero time reserve)
 * Orange elements: Near-critical activities
 * Black elements: Non-critical activities
 * Dashed lines: Dummy activities

