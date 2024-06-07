# Cylindrical Texture Calibrator

Astronomy-focused Python tool with a GUI to prepare texture maps of celestial bodies.
The main functionality is albedo calibration and gamma correction.

The program may be considered as a complement to the capabilities of [TrueColorTools](https://github.com/Askaniy/TrueColorTools).


## Installation

Python version 3.9 or higher is required. On Linux, you may need to replace the `python` command with `python3`.

**Step Zero**: Clone the repository or download the archive using the GitHub web interface. In the console, go to the project root folder.

### Simple installation
1. Install the dependencies with `pip install -r requirements.txt`;
2. Execute `python -u runCTC.py`.

### In virtual environment
1. Create a virtual environment with `python -m venv .venv`;
2. Install the dependencies with `.venv/bin/pip install -r requirements.txt`;
3. Execute `.venv/bin/python -u runCTC.py`.
