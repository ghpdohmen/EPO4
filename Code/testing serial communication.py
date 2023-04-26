# making a script to control the car using wasd-keys
import serial as serial


comport = 'COM3'
baud_rate = 115200
serial_port = serial.Serial(comport, baud_rate, rtscts=True)


def car_control(key_input):
    if key_input == "q":
        print("starting connection")
    elif key_input == "e":
        serial_port.close()
        print("stopping connection")
    elif key_input == "w":
        serial_port.write(b'M165\n')
        print("going forward")
    elif key_input == "a":
        serial_port.write(b'D200\n')
        print("going left")
    elif key_input == "s":
        serial_port.write(b'M135\n')
        print("going back")
    elif key_input == "d":
        serial_port.write(b'D100\n')
        print("going right")
    elif key_input == ".":
        exit()
    else:
        serial_port.write(b'M150\n')
        serial_port.write(b'D150\n')
        print("stopping car")


while True:
    car_control(input("enter key:"))
