from robot import Robot

# needs to be completely rewritten for GUI
b = Robot(0, 0)
b.start()
while True:
    b.update()
    if (input('test:') == '.'):
        b.stop()
