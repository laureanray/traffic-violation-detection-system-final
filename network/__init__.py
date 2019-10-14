import http.client
import json
import socketio
import threading, time  

class Net:
  def __init__(self):
    self.sio = socketio.Client()
    self.sio.connect('http://localhost:5000')
    self.connection = http.client.HTTPConnection('localhost:5000')
    self.headers = {'Content-type': 'application/json'}
    
  def sendRoutineUpdate(self, car_count, violation_count):
    payload = {'cars_detected': car_count, 'violations_detected': violation_count}
    json_payload = json.dumps(payload)
    self.connection.request('POST', '/routine_update', json_payload, self.headers)
    response = self.connection.getresponse()
    self.sio.emit('update', {'message': 'update'})


  def closeConnection(self):
    self.connection.close()
    self.sio.disconnect()
    