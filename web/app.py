from flask import Flask, render_template, send_from_directory, request
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["tvds-dev"]
col = db["violations"]
a = {"violation_type": "LEFT_TURN", "plate_number": "ABC-123"}

app = Flask(__name__)

x = col.insert_one(a)


@app.route('/')
def index():
    return render_template('template.html', page='index')


@app.route('/add_violation', methods=['POST'])
def add_violation():
    if request.is_json:
        data = request.get_json()
        print(data['test'])
    return 'JSON ok'


if __name__ == '__main__':
    app.run(debug=True)
