import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import ttk

from misc.robotModeEnum import robotMode

root = Tk()
root.title("GUI for EPO4")

window = ttk.Frame(root)

# graph showing path
figure = plt.Figure(figsize=(2,2), dpi=100)
a = figure.add_subplot(111)
a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])
canvas = FigureCanvasTkAgg(figure)
canvas.draw()
# end graph

# listbox, selecting different programs
def program_selector():
    selection = lb_programs.curselection()
    step_0 = str(selection).split(',')
    step_1 = step_0[0].split('(')
    step_2 = step_1[1]
    print(robotMode(int(step_2)))



l_programs = ["Manual", "Challenge A", "Challenge B", "Challenge C", "Challenge D", "Challenge E"]
l_programsvar = StringVar(value=l_programs)
lb_programs = Listbox(root, height=5, listvariable=l_programsvar)

# end listbox

# wasd key functionality
def m_forward():
    speed["value"] += 10

def m_right():
    pass

def m_left():
    pass

def m_backward():
    pass

button_w = ttk.Button(root, text="Forward", command=m_forward)
button_a = ttk.Button(root, text="Left", command=m_left)
button_s = ttk.Button(root, text="Backward", command=m_backward)
button_d = ttk.Button(root, text="Right", command=m_right)

root.bind("<w>", lambda event:m_forward())
root.bind("<a>", lambda event:m_left())
root.bind("<s>", lambda event:m_backward())
root.bind("<d>", lambda event:m_right())
# end wasd key

# speed bar
speed = ttk.Progressbar(root, orient=HORIZONTAL, length=200, maximum=30, mode="determinate")

#end speed bar

# grid for everything
canvas.get_tk_widget().grid(column=0, row=0, columnspan=3)

button_w.grid(column=1, row=4)
button_a.grid(column=0, row=5)
button_s.grid(column=1, row=5)
button_d.grid(column=2, row=5)

lb_programs.grid(column=3, row=0)

speed.grid(column=0, columnspan=3)
# end grid

root.mainloop()