import tkinter as tk
from tkinter import messagebox
import joblib

# Load trained model
model = joblib.load("spam_model.pkl")


def check_message():
    message = message_box.get("1.0", tk.END).strip()

    if message == "":
        messagebox.showwarning("Warning", "Please enter a message!")
        return

    prediction = model.predict([message])[0]

    if prediction == 1:
        result_label.config(
            text="🚨 SPAM MESSAGE",
            fg="red"
        )
    else:
        result_label.config(
            text="✅ NOT SPAM (HAM)",
            fg="green"
        )


def clear_message():
    message_box.delete("1.0", tk.END)
    result_label.config(text="Result will appear here", fg="black")


# Main window
root = tk.Tk()
root.title("Spam Email Classifier")
root.geometry("650x500")
root.resizable(False, False)

# Heading
title_label = tk.Label(
    root,
    text="📧 Spam Email Classifier",
    font=("Arial", 24, "bold")
)
title_label.pack(pady=25)

# Description
description = tk.Label(
    root,
    text="Enter a message below to check whether it is Spam or Not Spam",
    font=("Arial", 11)
)
description.pack(pady=5)

# Message box
message_box = tk.Text(
    root,
    height=8,
    width=65,
    font=("Arial", 12)
)
message_box.pack(pady=20)

# Check button
check_button = tk.Button(
    root,
    text="🔍 Check Message",
    command=check_message,
    font=("Arial", 12, "bold"),
    padx=20,
    pady=10
)
check_button.pack(pady=5)

# Clear button
clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_message,
    font=("Arial", 10),
    padx=20,
    pady=5
)
clear_button.pack(pady=5)

# Result
result_label = tk.Label(
    root,
    text="Result will appear here",
    font=("Arial", 18, "bold")
)
result_label.pack(pady=25)

# Start application
root.mainloop()