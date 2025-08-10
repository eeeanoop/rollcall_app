# Automated Roll Call

This is a Python application that uses facial recognition for an automated roll call system. It's built with PySide6 for the GUI and InsightFace for the core facial recognition logic.

## Features

- Live camera feed for real-time recognition.
- Roster building from a folder of student images.
- Identifies present, absent, and unauthorized individuals.
- Saves session results to a JSON file.
- Audio feedback for recognized and unauthorized persons.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:eeeanoop/rollcall_app.git
    cd rollcall_app
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. See below.)*

3.  **Add student images:**
    Place images of students in the `student_images` directory. The filename (without extension) will be used as the student's name (e.g., `John_Doe.jpg`).

## Usage

Run the application with:
```bash
python3 app.py
```