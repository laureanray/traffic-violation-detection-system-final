import http.client
import json
import socketio

sio = socketio.Client()

sio.connect('http://localhost:5000')

connection = http.client.HTTPConnection('localhost:5000')
headers = {'Content-type': 'application/json'}

payload = {'cars_detected': 2}
json_payload = json.dumps(payload)

connection.request('POST', '/update_car_count', json_payload, headers)

response = connection.getresponse()

sio.emit('routine', {'foo': 'bar'})

print(response.read().decode())
