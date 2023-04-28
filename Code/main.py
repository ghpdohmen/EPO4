from robot import Robot

# TODO: implement GUI
b = Robot(0, 0)
b.start()
while True:
    b.update()
    if input('test:') == '.':
        b.stop()
