"""
action_mappers.py

Maps gesture labels from your trained model to application/system actions.
Uses pyautogui for simulating keypresses when available.
Add or customize actions here.
"""

import time
try:
    import pyautogui
    PYA_AVAILABLE = True
except Exception:
    PYA_AVAILABLE = False
    print("pyautogui not available; actions will print to console instead.")

class ActionMapper:
    def __init__(self):
        # Cooldown to prevent repeated activations (seconds)
        self.cooldown = 1.0
        self.last_trigger = {}

        # Map gesture labels to actions
        self.map = {
            '01_palm': self.play_pause,
            '03_fist': self.prev_slide,
            '06_index': self.next_slide,
            '07_ok': self.open_app,
            '10_down': self.volume_down,
            '09_c': self.volume_up
            # Add more gestures here if needed
        }

    def trigger(self, label):
        now = time.time()
        last = self.last_trigger.get(label, 0)
        if now - last < self.cooldown:
            return  # Still in cooldown

        if label not in self.map:
            print(f"No action mapped for {label}")
            return

        try:
            self.map[label]()
            self.last_trigger[label] = now
        except Exception as e:
            print("Error executing action:", e)

    # -------------------------------
    # Example Actions
    # -------------------------------
    def volume_up(self):
        if PYA_AVAILABLE:
            pyautogui.press('volumeup') if hasattr(pyautogui, 'press') else None
        print("Action: Volume Up")

    def volume_down(self):
        if PYA_AVAILABLE:
            pyautogui.press('volumedown') if hasattr(pyautogui, 'press') else None
        print("Action: Volume Down")

    def play_pause(self):
        if PYA_AVAILABLE:
            pyautogui.press('space')
        print("Action: Play/Pause")

    def next_slide(self):
        if PYA_AVAILABLE:
            pyautogui.press('right')
        print("Action: Next Slide")

    def prev_slide(self):
        if PYA_AVAILABLE:
            pyautogui.press('left')
        print("Action: Previous Slide")

    def open_app(self):
        # Placeholder for launching an app
        print("Action: Open App (no-op)")
