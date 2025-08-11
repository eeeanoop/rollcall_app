# Automated Roll Call (Ubuntu Setup Guide)

This is a Python application that uses facial recognition for an automated roll call system. It's built with PySide6 for the GUI and InsightFace for the core facial recognition logic. This guide is specifically for users on Ubuntu Linux.

## Features

- Live camera feed for real-time recognition.
- Roster building from a folder of student images.
- Identifies present, absent, and unauthorized individuals.
- Saves session results to a JSON file.
- Audio feedback for recognized and unauthorized persons.

## Part 1: System Prerequisites

Before setting up the application, you need to ensure your system has the necessary tools.

### Step 1: Install Python and Essential Tools
Most versions of Ubuntu come with Python 3, but you'll need `pip` (the Python package installer), `venv` (for creating virtual environments), and `unzip`.

Open a terminal (you can use the shortcut `Ctrl+Alt+T`) and run the following commands:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv unzip
```

### Step 2: Install Application Dependencies
The application requires a text-to-speech engine for audio feedback and a math library for better performance.

```bash
sudo apt install -y espeak-ng libatlas-base-dev
```

## Part 2: Application Setup

Now that your system is ready, you can set up the RollCall application.

### Step 1: Download and Unzip the Code
You can download the project code directly from GitHub as a ZIP file without using Git.

1.  Go to the repository page: https://github.com/eeeanoop/rollcall_app
2.  Click the green **<> Code** button.
3.  Select **Download ZIP**.
4.  Once downloaded, find the `rollcall_app-main.zip` file in your `Downloads` folder. Unzip it using your file manager or by running this command in the terminal:
    ```bash
    unzip ~/Downloads/rollcall_app-main.zip -d ~/
    cd ~/rollcall_app-main
    ```
    *Note: The unzipped folder will likely be named `rollcall_app-main`. We will work inside this folder for the next steps.*

### Step 2: Create a Virtual Environment and Install Python Packages
It's a best practice to keep project dependencies isolated from your system's Python packages.

```bash
# From inside the project folder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*You'll need to run `source venv/bin/activate` every time you open a new terminal to work on this project.*

### Step 4: Add Student Images
Place images of students in the `student_images` directory. The filename (without the extension) will be used as the student's name (e.g., `anoop.jpg`).

## Part 3: Running the Application

With everything set up, you can now run the application.

```bash
# Make sure your virtual environment is activated
python3 app.py
```
