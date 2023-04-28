import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import ttk

import robot
from misc.robotModeEnum import robotMode


class App(tk.Frame):
    enableRobot = False

    # wasd key functionality
    def m_forward(self):
        self._robot.inputSubSystem.keyboard_w()
        pass

    def m_right(self):
        self._robot.inputSubSystem.keyboard_d()
        pass

    def m_left(self):
        self._robot.inputSubSystem.keyboard_a()
        pass

    def m_backward(self):
        self._robot.inputSubSystem.keyboard_s()
        pass

    def estop(self):
        self._robot.inputSubSystem.estop()

    def startRobot(self):
        self._robot.start()
        self.enableRobot = True

    def updateRobot(self):
        if self.enableRobot:
            self._robot.update()
        # self.after(100, self.updateRobot())

    def stopRobot(self):
        self._robot.stop()
        self.enableRobot = False

    def __init__(self):
        self._robot = robot.Robot(0, 0)


a = App()

root = Tk()
root.title("GUI for EPO4")
window = ttk.Frame(root)

# graph showing path
figure = plt.Figure(figsize=(2, 2), dpi=100)
axis = figure.add_subplot(111)
axis.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])
canvas = FigureCanvasTkAgg(figure)
canvas.draw()


# end graph

# robot declaration

# listbox, selecting different programs
def program_selector():
    _selection = lb_programs.curselection()
    _step_0 = str(_selection).split(',')
    _step_1 = _step_0[0].split('(')
    _step_2 = _step_1[1]
    print(robotMode(int(_step_2)))


l_programs = ["Manual", "Challenge A", "Challenge B", "Challenge C", "Challenge D", "Challenge E"]
l_programsvar = StringVar(value=l_programs)
lb_programs = Listbox(root, height=5, listvariable=l_programsvar)
lb_programs.bind("<Double-Button-1>", lambda event: program_selector())
# end listbox

# buttons
# start button
button_start = ttk.Button(root, text="START", command=a.startRobot)

button_stop = ttk.Button(root, text="DISABLE ROBOT", command=a.stopRobot)

button_estop = ttk.Button(root, text="STOP MOTORS", command=a.estop)

# bindings
root.bind("<.>", lambda event: a.estop())
root.bind("<w>", lambda event: a.m_forward())
root.bind("<a>", lambda event: a.m_left())
root.bind("<s>", lambda event: a.m_backward())
root.bind("<d>", lambda event: a.m_right())
# end buttons



# speed bar
speed = ttk.Progressbar(root, orient=HORIZONTAL, length=200, maximum=30, mode="determinate")

# end speed bar

# grid for everything
canvas.get_tk_widget().grid(column=0, row=0, columnspan=3, rowspan=3)

button_start.grid(column=3, row=0)
button_stop.grid(column=3, row=1)
button_estop.grid(column=3, row=2)


lb_programs.grid(column=3, row=4)

speed.grid(column=0, columnspan=3)
# end grid

while True:
    a.updateRobot()
    root.update()

root.mainloop()
