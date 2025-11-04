# Infinity-Synth: High-Quality Synthetic Document Data Generation

## Quick Start
  
### 🧭 Step 1: Google Chrome Headless Setup

This document provides instructions for checking, installing, and running Google Chrome in headless mode — useful for web automation, screenshots, PDF rendering, or server-side rendering tasks.

#### 1. Check Installed Chrome Version

You can verify if Chrome (or Chromium) is already installed and check its version by running:

```shell
google-chrome --version
```
or

```shell
chromium-browser --version
```

#### 2. Install Google Chrome (Ubuntu Example)

```shell
# Update package index
sudo apt-get update
# Install dependencies
sudo apt-get install -y libappindicator1 fonts-liberation
# Download Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# Install the package
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt --fix-broken install
# Verify installation
google-chrome --version
```

#### 3. Please download Chromedriver, place it in the drive directory, name it chromedriver, and grant it execution permission.
    
### 🚀 Step 2: Run Data Synthesis

```shell
python main.py --config=examples/three_columns.yaml
```

### 🧩 Step 3: Convert Synthesized Data into Markdown

```shell
python scripts/doc_parser.py --config=examples/three_columns.yaml
```
📁 The synthesized data will be saved in `results.json`.  
You can modify the save path by updating `work_path.result` in `examples/three_columns.yaml`.


### 🛠️ Optional: Extending Template and Style Diversity
If you want to add new layout styles, modify the template specified by `work_path.template_file` and the corresponding data-filling function defined in `work_path.template_get_data`.  
These control the structure and content generation logic of the synthetic samples.  
For additional customization, please refer to the following parameters.

```
data_paths:
  text: "examples/data/text.json"
  image: "examples/data/figure.json"
  table: "examples/data/table.json"
  formula: "examples/data/formula.json"
  title: ""
```  

```
work_path:
  template_path: "templates"
  template_file: "three_columns/document.html.jinja"
  template_get_data: "three_columns/getData"
  html_path: "/path/to/Infinity_Synth/working/html/output_{i}.html"
  save_image_dir: "working/image/"
  output_gt_path: "working/ground_truth/result_of_id{i}.json"
```

> Important: Always provide an absolute path for `html_path`

- save_image_dir: Directory path where the final images of rendered HTML pages will be stored.

```
defaults:
  save_path: "Temp"
  work_path_template: "Temp_process_id{process_id}"
  output_file_template: "result_of_id{process_id}.json"
  save_every_n: 40
```

```
layout_config:
  element:
    table: 1
    figure: 1
    title: 0
    text: 6
    formula: 3
    header: 1
    footer: 1
    page_footnote: 1
  columns: 1
```

- element: defines the **maximum** number of elements for a single page.
- columns: the number of columns. Now only support 1.

```
num_workers: 10
nums: 1000
```
- num_workers: The number of parallel workers/processes to be used.

- nums: The total number of data samples to be processed.
