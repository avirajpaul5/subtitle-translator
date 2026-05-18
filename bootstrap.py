#!/usr/bin/env python3
"""Shown while PySide6 is being installed on first run (stdlib only)."""
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Subtitle Translator")
root.geometry("460x150")
root.resizable(False, False)

root.update_idletasks()
x = (root.winfo_screenwidth() - 460) // 2
y = (root.winfo_screenheight() - 150) // 2
root.geometry(f"460x150+{x}+{y}")

tk.Label(
    root,
    text="Setting up Subtitle Translator…",
    font=("Helvetica", 14, "bold"),
).pack(pady=(22, 4))

tk.Label(
    root,
    text="Installing PySide6 on first run (~100 MB). This takes about 30–60 seconds.",
    font=("Helvetica", 11),
    fg="#555555",
).pack()

bar = ttk.Progressbar(root, mode="indeterminate", length=400)
bar.pack(pady=14)
bar.start(10)

root.mainloop()
