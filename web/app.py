from flask import Flask, render_template, send_from_directory, request
import pymongo
import json 
from flask import jsonify
from flask_socketio import SocketIO, send, emit
from bson import Binary, Code
from bson.json_util import dumps
import threading, time

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["tvds-dev"]
violations_collection = db["violations"]
global_collection = db['global']

app = Flask(__name__)

socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('template.html', page='index')


# @app.route('/add_violation', methods=['POST'])
# def add_violation():
#     if request.is_json:
#         data = request.get_json()
#         print(data['test'])
#     return 'JSON ok'

# a = global_collection.find_one()

# print(a)
@app.route('/get_global', methods=['GET'])
def get_global():
    a = global_collection.find_one()
    del a['_id']
    print(a)
    return jsonify(a)

@app.route('/update_car_count', methods=['POST'])
def update_num_car():
    if request.is_json:
        data = request.get_json()
        cars = data['cars_detected']
        query = {'_Q': 'global'}
        values = { '$set': { 'cars_detected': cars}}
        global_collection.update_one(query, values)
        
        
    res = {'response': 'ok'}
    # json_res = json.dumps(res)
    return jsonify(res)

@app.route('/add_violation', methods=['POST'])
def add_violation():
    if request.is_json:
        data = request.get_json()
        violation_type = data['violation_type']
        plate_number = data['plate_number']
        plate_number_img_url = data['plate_number_img_url']
        vehicle_img_url = data['vehicle_img_url']
        vehicle_type = data['vehicle_type']
        
        data_to_insert = {
            'violation_type': violation_type,
            'vehicle_type': vehicle_type,
            'plate_number': plate_number,
            'plate_number_img_url': plate_number_img_url,
            'vehicle_img_url': vehicle_img_url            
        }
        
        violations_collection.insert_one(data_to_insert)
        
        return jsonify({'res': 'ok'})



def initialize_databse():
    
    data = {
        '_Q': 'global',
        'cars_detected': 0,
        'violation_detected': 0
    }
    
    # print(current_col.count())
    
    if global_collection.count() == 0:
        res = global_collection.insert_one(data)
    
    print('Database Initialized')


@socketio.on('update')
def handle_message(message):
    emit('update', {'message': message}, broadcast=True)    
    # print('received message: ' + message['foo'])
    
    
# @socketio.on('connected')
# def handle_connect():
#     print('connectedasdasd')
#     emit('message', {'data': 'test'}, broadcast=True)    
    
# This ensures that database contains a single document for global state   
initialize_databse()


if __name__ == '__main__':
    socketio.run(app)



def runServer():
    socketio.run(app)


def runServerOnThread():    
    thread = threading.Thread(target = runServer)
    thread.start();