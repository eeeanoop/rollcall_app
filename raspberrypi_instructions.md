
# 🚀 Setting Up Your Raspberry Pi for the RollCall App

Welcome! This guide will walk you through setting up your Raspberry Pi to run the automated RollCall application. Follow these steps, and you'll be up and running in no time.

---

## 🛒 Part 1: What You'll Need (The Hardware)

Here’s a list of recommended parts to build your facial recognition station.

*   **🤖 Compute:**
    *   **Raspberry Pi 5 (8 GB RAM recommended):** This is the brain of our project! The 4 GB version works, but 8 GB gives you extra power so things run smoothly without slowing down.

*   **💾 Storage:**
    *   **32-64 GB microSD Card (A2-rated):** A fast (A2-rated) card is important for the operating system to run quickly.
    *   *(Optional Upgrade)*: An NVMe SSD with the Pi 5 M.2 HAT+ will make everything even faster!

*   **⚡️ Power:**
    *   **Official 27W USB-C Power Supply:** The Pi 5 needs a lot of power. Using the official supply prevents random shutdowns and performance issues.

*   **💨 Cooling:**
    *   **Active Cooler (Case with fan or official fan):** The Pi 5 can get hot when it's thinking hard (like during facial recognition!). A fan is essential to prevent it from overheating and slowing down.

*   **📸 Camera (Choose one):**
    *   **Pi Camera Module 3:** Connects directly to the Pi for a super sharp, low-latency video feed.
    *   **A good USB Webcam:** Any webcam that supports 1080p at 30fps will also work great. Just plug it into a USB port.

*   **🔊 Audio Output:**
    *   **Small USB Speaker, HDMI Monitor with Speakers, or 3.5mm Speaker:** This is so the app can talk to you and announce who it sees!

*   **🔩 Mounting:**
    *   **Mini Tripod or a Doorframe Mount:** You'll want to position the camera at about head height so it can easily see everyone's faces.

---

## ⚙️ Part 2: Step-by-Step Software Setup

Now let's get the software installed and configured.

### 1. Flash the Operating System (OS)
The OS is the base software that runs the Pi. We need a specific version for the best performance.

*   **OS Choice:** **Raspberry Pi OS (64-bit) with Desktop**.
*   **Why 64-bit?** The facial recognition software we're using runs much faster on a 64-bit system. The 32-bit version will be very slow!
*   **How:** Use the [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash the OS onto your microSD card.

### 2. First Boot and System Updates
Plug everything into your Pi and turn it on! Once you're at the desktop:

*   Open a **Terminal** window.
*   Run the following command to make sure all your software is up-to-date. This might take a few minutes.
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

### 3. Install Essential Tools
Our project needs a couple of extra tools to work correctly.

*   `espeak-ng`: This is a text-to-speech engine. It lets the Pi talk!
*   `libatlas-base-dev`: This is a math library that helps speed up calculations for facial recognition.

*   Install them with this command in the Terminal:
    ```bash
    sudo apt update && sudo apt install -y espeak-ng libatlas-base-dev
    ```

### 4. Get the Project Code
Now, let's download the RollCall application code from its repository.

*   In your Terminal, run:
    ```bash
    git clone <your-repository-url>
    cd automated-roll-call
    ```
    *(Note: Replace `<your-repository-url>` with the actual URL to the project's Git repository.)*

### 5. Set Up the Python Environment
We'll create a "virtual environment" to keep our project's Python packages separate from the system's. This is a best practice to avoid conflicts.

*   **Create the environment:**
    ```bash
    python3 -m venv venv
    ```
*   **Activate it:** (You'll need to do this every time you open a new terminal to work on the project)
    ```bash
    source venv/bin/activate
    ```
*   **Install the Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### 6. Configure the Camera
Let's make sure the Raspberry Pi knows you've connected a camera.

*   **If you are using a USB Webcam:** It should work automatically! You can skip to the next step.
*   **If you are using a Pi Camera Module (CSI):**
    1.  Open the Terminal and run `sudo raspi-config`.
    2.  Navigate to `3 Interface Options`.
    3.  Select `I1 Legacy Camera` and enable it.
    4.  The tool will ask you to reboot. Select **Yes**.

### 7. Add Student Photos
The app needs to know who to look for!

*   Find the `student_images/` folder inside the project directory.
*   Drop your student photos in here. Make sure each photo is a clear shot of one person's face.
*   Rename the files to the student's name, like `Grace_Hopper.jpg` or `Alan_Turing.png`. The filename is used as their name in the app.

### 8. You're Ready to Go!
That's it for the setup. To start the automated roll call:

*   Make sure you are in the project directory (`automated-roll-call`) and your virtual environment is activated.
*   Run the application:
    ```bash
    python3 app.py
    ```

Enjoy your new automated roll call system! 🎉