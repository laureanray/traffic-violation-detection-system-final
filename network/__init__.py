import http.client
import json
import socketio

sio = socketio.Client()

sio.connect('http://localhost:5000')

connection = http.client.HTTPConnection('localhost:5000')
headers = {'Content-type': 'application/json'}


def sendRoutineUpdate(car_count):
  payload = {'cars_detected': car_count}
  json_payload = json.dumps(payload)
  connection.request('POST', '/update_car_count', json_payload, headers)
  response = connection.getresponse()
  sio.emit('update', {'message': 'update'})


def closeConnection():
  connection.close()
  sio.disconnect()
# @sio.on('message')
# def on_message(data):
#     print('I received a message!')
#     print(data)
