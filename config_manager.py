from application import State

def openFileWriteMode():
    State.config = open(State.config_path, "w+")
    print('write mode')

def openFileReadMode():
    State.config = open(State.config_path, "r+")
    print('read mode')

def loadConfig():
    data = []
    # Open config as read mode
    openFileReadMode()

    with State.config as file:
        data = file.readlines()

    for item in data:
        key_value = item.split(' ')
        State.config_dict[key_value[0]] = key_value[1].rstrip()
        
def saveConfig(data):
    camera1 = data['camera1']
    camera2 = data['camera2']
    plate_inference_graph = data['plate_inference']
    car_inference_graph = data['car_inference']
    log = data['log']

