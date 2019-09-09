from application import State

def open_file_writemode():
    State.config = open(State.config_path, "w+")
    print('write mode')

def open_file_readmode():
    State.config = open(State.config_path, "r+")
    print('read mode')

def load_config():
    data = []
    # Open config as read mode
    open_file_readmode()

    with State.config as file:
        data = file.readlines()

    for item in data:
        key_value = item.split(' ')
        State.config_dict[key_value[0]] = key_value[1].rstrip()

