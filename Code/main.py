import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import ttk

import robot
from misc.robotModeEnum import robotMode

#This is the main program file for the EPO-4 project. This code handles the GUI and instatiates the robot, from which the entire program functions.


# ----------------------------------------------------------------------------------------------------------------------
# functions and initialisation                                          

class App(tk.Frame):
    enableRobot = False
    selectedRobotMode = robotMode.NotChosen

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
        self._robot.start(self.selectedRobotMode)
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



# ----------------------------------------------------------------------------------------------------------------------
# windows to display data

# little window to show data
window = ttk.Frame(root, width=400, height=200)
window["borderwidth"] = 4
window["relief"] = "ridge"
# end window

# little window for subsystem status
ssWindow = ttk.Frame(root, width=400, height=200)
ssWindow["borderwidth"] = 4
ssWindow["relief"] = "ridge"


challengeWindow = ttk.Frame(root, width=400, height=200)
challengeWindow["borderwidth"] = 4
challengeWindow["relief"] = "ridge"

# ----------------------------------------------------------------------------------------------------------------------
# graphing

# TODO: show robot location, location uncertainty, goal and path? on this plot
# graph showing path
figure = plt.Figure(figsize=(3, 3), dpi=100)
fig = figure.add_subplot(111, autoscale_on=False, xlim=(0, 480), ylim=(0, 480))
fig.set_aspect('equal')
fig.grid()
canvas = FigureCanvasTkAgg(figure)
canvas.draw()
plot_button = tk.Button(text="plot", command=lambda: fig_plot(canvas))

def fig_plot(canvas):
    fig.clear()
    _start = robot.Robot.startPos
    _a_end = robot.Robot.aEnd
    _b_mid = robot.Robot.bMid
    _b_end = robot.Robot.bEnd
    _x = np.array([0, 1, 2, 3, 4, 5])
    _y = np.array([5, 4, 3, 2, 1, 0])
    fig.plot(_x, _y)
    fig.plot(_start[0], _start[1], marker='o', markeredgecolor='red', markerfacecolor='white')
    fig.plot(_a_end[0], _a_end[1], marker='o', markeredgecolor='red', markerfacecolor='white')
    fig.plot(_b_mid[0], _b_mid[1], marker='o', markeredgecolor='red', markerfacecolor='white')
    fig.plot(_b_end[0], _b_end[1], marker='o', markeredgecolor='red', markerfacecolor='white')
    # add sum to limit axes to 0 x 480
    canvas.draw()
# end graph



# ----------------------------------------------------------------------------------------------------------------------
# selecting programs

# listbox, selecting different programs
def program_selector():
    _selection = lb_programs.curselection()
    _step_0 = str(_selection).split(',')
    _step_1 = _step_0[0].split('(')
    _step_2 = _step_1[1]
    # print(robotMode(int(_step_2)))
    a.selectedRobotMode = robotMode(int(_step_2))

l_programs = ["Manual", "Challenge A", "Challenge B", "Challenge C", "Challenge D", "Challenge E"]
l_programsvar = StringVar(value=l_programs)
lb_programs = Listbox(root, height=5, listvariable=l_programsvar)
lb_programs.bind("<Double-Button-1>", lambda event: program_selector())
# end listbox



# ----------------------------------------------------------------------------------------------------------------------
# updating comports

# comport updater:
def comport_updater():
    _number = comport_text.get('1.0', 'end')
    _comport = "COM"+str(_number)
    robot.Robot.COMport = _comport.split('\n')[0]
    print(robot.Robot.COMport)

comport_text = Text(root, width=2, height=1)
comport_text.insert('1.0', '4')

# ----------------------------------------------------------------------------------------------------------------------
# updating challenge locations

# challenge locations:
def location_updater():
    _start = ([int(start_x.get('1.0', 'end').split('\n')[0]), int(start_y.get('1.0', 'end').split('\n')[0])])
    _end_a = ([int(end_a_x.get('1.0', 'end').split('\n')[0]), int(end_a_y.get('1.0', 'end').split('\n')[0])])
    _mid_b = ([int(mid_b_x.get('1.0', 'end').split('\n')[0]), int(mid_b_y.get('1.0', 'end').split('\n')[0])])
    _end_b = ([int(end_b_x.get('1.0', 'end').split('\n')[0]), int(end_b_y.get('1.0', 'end').split('\n')[0])])
    
    robot.Robot.startPos = _start
    robot.Robot.aEnd = _end_a
    robot.Robot.bMid = _mid_b
    robot.Robot.bEnd = _end_b
    
    print(robot.Robot.startPos, _end_a, _mid_b, _end_b)
    

start_x = Text(challengeWindow, width=3, height=1)
start_x.insert('1.0', '0')
start_y = Text(challengeWindow, width=3, height=1)
start_y.insert('1.0', '0')  

end_a_x = Text(challengeWindow, width=3, height=1)
end_a_x.insert('1.0', '0')
end_a_y = Text(challengeWindow, width=3, height=1)
end_a_y.insert('1.0', '0')

mid_b_x = Text(challengeWindow, width=3, height=1)
mid_b_x.insert('1.0', '0')
mid_b_y = Text(challengeWindow, width=3, height=1)
mid_b_y.insert('1.0', '0')

end_b_x = Text(challengeWindow, width=3, height=1)
end_b_x.insert('1.0', '0')
end_b_y = Text(challengeWindow, width=3, height=1)
end_b_y.insert('1.0', '0')


# ----------------------------------------------------------------------------------------------------------------------
# labels

label_0 = ttk.Label(window, text="Selected program:")
label_mode = ttk.Label(window)
operatingModeText = StringVar()
label_mode["textvariable"] = operatingModeText
operatingModeText.set(str(a.selectedRobotMode).split('.')[1])

label_1 = ttk.Label(window, text="speed: ")
label_m = ttk.Label(window)
motorSpeedText = StringVar()
label_m["textvariable"] = motorSpeedText
motorSpeedText.set(robot.Robot.input_motor)

label_2 = ttk.Label(window, text="battery: ")
label_b = ttk.Label(window)
batteryText = StringVar()
label_b["textvariable"] = batteryText
batteryText.set(robot.Robot.batteryVoltage)

label_3 = ttk.Label(window, text="direction: ")
label_d = ttk.Label(window)
directionText = StringVar()
label_d["textvariable"] = directionText
directionText.set(robot.Robot.input_servo)

label_4 = ttk.Label(window, text="update frequency: ")
label_f = ttk.Label(window)
frequencyText = StringVar()
label_f["textvariable"] = frequencyText
frequencyText.set(str(np.round((1 / robot.Robot.averageLoop) * 10) / 10) + " Hz")
# end labels

# subsystem status labels
# main window title
label_SSStatuswindow = ttk.Label(ssWindow, text="Subsystem status", font=("Arial", 15))
# coms
label_coms = ttk.Label(ssWindow, text="Communications:")
label_comsStatus = ttk.Label(ssWindow)
comsStatusText = StringVar()
label_comsStatus["textvariable"] = comsStatusText
comsStatusText.set(str(robot.Robot.communicationState).split('.')[1])
# input
label_input = ttk.Label(ssWindow, text="Input:")
label_inputStatus = ttk.Label(ssWindow)
inputStatusText = StringVar()
label_inputStatus["textvariable"] = inputStatusText
inputStatusText.set(str(robot.Robot.inputState).split('.')[1])
# time
label_time = ttk.Label(ssWindow, text="Time:")
label_timeStatus = ttk.Label(ssWindow)
timeStatusText = StringVar()
label_timeStatus["textvariable"] = timeStatusText
timeStatusText.set(str(robot.Robot.timingState).split('.')[1])
# logging
label_logging = ttk.Label(ssWindow, text="Logging:")
label_loggingStatus = ttk.Label(ssWindow)
loggingStatusText = StringVar()
label_loggingStatus["textvariable"] = loggingStatusText
loggingStatusText.set(str(robot.Robot.loggingState).split('.')[1])
# localization
label_localization = ttk.Label(ssWindow, text="Localization:")
label_localizationStatus = ttk.Label(ssWindow)
localizationStatusText = StringVar()
label_localizationStatus["textvariable"] = localizationStatusText
localizationStatusText.set(str(robot.Robot.localizationState).split('.')[1])

# For challenge box
robot_start = ttk.Label(challengeWindow, text="Start:")
a_end = ttk.Label(challengeWindow, text="End A:")
b_mid = ttk.Label(challengeWindow, text="Mid B:")
b_end = ttk.Label(challengeWindow, text="End B:")



# ----------------------------------------------------------------------------------------------------------------------
# buttons

# start button
button_start = ttk.Button(root, text="START", command=a.startRobot)

# stop button
button_stop = ttk.Button(root, text="DISABLE ROBOT", command=a.stopRobot)

# emergency motor stop
button_estop = ttk.Button(root, text="STOP MOTORS", command=a.estop)

#comport button
comport_button = ttk.Button(root, text="SET COMPORT", command=comport_updater)

# location accept
location_button = ttk.Button(challengeWindow, text="ACCEPT", command=location_updater)

# bindings
root.bind("<.>", lambda event: a.estop())
root.bind("<w>", lambda event: a.m_forward())
root.bind("<a>", lambda event: a.m_left())
root.bind("<s>", lambda event: a.m_backward())
root.bind("<d>", lambda event: a.m_right())

# end buttons



# ----------------------------------------------------------------------------------------------------------------------
# grid

#grid for everything
window.grid(column=0, row=4)
ssWindow.grid(column=4, row=0, rowspan=2)
challengeWindow.grid(column=4, row=2, rowspan=2)

# subsystem status window
label_SSStatuswindow.grid(column=0, row=0, columnspan=2)
label_coms.grid(column=0, row=1)
label_comsStatus.grid(column=2, row=1)
label_input.grid(column=0, row=2)
label_inputStatus.grid(column=2, row=2)
label_time.grid(column=0, row=3)
label_timeStatus.grid(column=2, row=3)
label_logging.grid(column=0, row=4)
label_loggingStatus.grid(column=2, row=4)
label_localization.grid(column=0, row=5)
label_localizationStatus.grid(column=2, row=5)

# data window
label_0.grid(column=0, row=0)
label_mode.grid(column=1, row=0)
label_1.grid(column=0, row=1)
label_m.grid(column=1, row=1)
label_2.grid(column=0, row=2)
label_b.grid(column=1, row=2)
label_3.grid(column=0, row=3)
label_d.grid(column=1, row=3)
label_4.grid(column=0, row=4)
label_f.grid(column=1, row=4)

# challenge location window
robot_start.grid(column=0, row=0)
a_end.grid(column=0, row=1)
b_mid.grid(column=0, row=2)
b_end.grid(column=0, row=3)
start_x.grid(column=1, row=0)
start_y.grid(column=2, row=0)
end_a_x.grid(column=1, row=1)
end_a_y.grid(column=2, row=1)
mid_b_x.grid(column=1, row=2)
mid_b_y.grid(column=2, row=2)
end_b_x.grid(column=1, row=3)
end_b_y.grid(column=2, row=3)
location_button.grid(column=0, row=4, columnspan=3)

#graph
canvas.get_tk_widget().grid(column=0, row=0, columnspan=3, rowspan=3)
plot_button.grid(column=1, row=4)

#buttons
button_start.grid(column=3, row=0)
button_stop.grid(column=3, row=1)
button_estop.grid(column=3, row=2)
comport_button.grid(column=4, row=4)


#selection
lb_programs.grid(column=3, row=4)

#comport text
comport_text.grid(column=5, row=4)

# end grid



# ----------------------------------------------------------------------------------------------------------------------
# update loops

def gui_updater():
    batteryText.set(robot.Robot.batteryVoltage)
    directionText.set(robot.Robot.input_servo)
    operatingModeText.set(str(a.selectedRobotMode).split('.')[1])
    localizationStatusText.set(str(robot.Robot.localizationState).split('.')[1])
    loggingStatusText.set(str(robot.Robot.loggingState).split('.')[1])
    timeStatusText.set(str(robot.Robot.timingState).split('.')[1])
    inputStatusText.set(str(robot.Robot.inputState).split('.')[1])
    comsStatusText.set(str(robot.Robot.communicationState).split('.')[1])
    motorSpeedText.set(robot.Robot.input_motor)
    frequencyText.set(str(np.round((1 / robot.Robot.averageLoop) * 10) / 10) + " Hz")

    # update robot and gui
    a.updateRobot()
    root.update()
    
    root.after(100, gui_updater) #updates itself after 100 ms

gui_updater()

root.mainloop()