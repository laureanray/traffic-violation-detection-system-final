from flask import Flask, render_template, send_from_directory, request
import pymongo
import json 
from flask import jsonify
from flask_socketio import SocketIO, send, emit
from bson import Binary, Code
from bson.json_util import dumps
import threading, time
from multiprocessing import Process
from network import Net

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["tvds-dev"]
violations_collection = db["violations"]
global_collection = db['global']
user_collection = db['user']

app = Flask(__name__)

socketio = SocketIO(app)



@app.route('/')
def index():
    return render_template('template.html', page='index')

@app.route('/get_global', methods=['GET'])
def get_global():
    a = global_collection.find_one()
    del a['_id']
    print(a)
    return jsonify(a)

@app.route('/routine_update', methods=['POST'])
def update_num_car():
    if request.is_json:
        data = request.get_json()
        cars = data['cars_detected']
        violations = data['violations_detected']
        query = {'_Q': 'global'}
        values = { '$set': { 'cars_detected': cars, 'violations_detected': violations}}
        
        global_collection.update_one(query, values)
        
        
    res = {'response': 'ok'}
    # json_res = json.dumps(res)
    return jsonify(res)

# @app.route('/add_violation', methods=['POST'])
def add_violation(data):
    # pritn('add violation called')
    # if request.is_json:
    with app.app_context():
        data_to_insert = {
            'violation_type': data['violation_type'],
            'vehicle_type': data['vehicle_type'],
            'plate_number': data['plate_number'],
            'plate_number_img_url': data['plate_number_img_url'],
            'vehicle_img_url': data['vehicle_img_url']            
        }
        
        violations_collection.insert_one(data_to_insert)
        
        return jsonify({'res': 'ok'})


@app.route('/get_violations', methods=['GET'])
def get_violations():
    res = []
    # data = violations_collection.find({})
    for x in violations_collection.find():
        del x['_id']
        res.append(x)
        
    print(res)
    
    return json.dumps(res)

def initialize_databse():
    
    data = {
        '_Q': 'global',
        'cars_detected': 0,
        'violation_detected': 0
    }
    
    adminData = {
        'username': 'admin',
        'password': 'P@$$w0rd'
    }
    # print(current_col.count())
    
    if global_collection.count() == 0:
        res = global_collection.insert_one(data)
        
    if user_collection.count() == 0:
        res2 = user_collection.insert_one(adminData)
    
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
    socketio.run(app, port=3001)



def runServer():
    socketio.run(app)

# thread = threading.Thread(target = runServer)
process = Process(target = runServer)


def runServerOnThread():    
    process.start()
    
def shutdownServerOnThread():
    process.terminate()