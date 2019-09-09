
class Details:
    width = 720
    height = 540
    name = "Traffic Violation Detection System"
    version = "1"


class State:
    footage = ""
    logging = None
    main = None
    dashboard = None
    model_path = ""
    pbtext_path = ""
    source = ""
    isStarted = False
    config_path = "config/application.config"
    config = None
    config_dict = {
        'CAMERA_1': '/',
        'CAMERA_2': '/'
    }

