from flask import Flask, request
from flask_cors import CORS
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from recourse_api import setup

# DOCS https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
executor = ThreadPoolExecutor(2)
app = Flask(__name__)
CORS(app)
setup_recourse = executor.submit(setup)

@app.route("/", methods=['GET'])
def index():
    return "<h1>This is Recourse API!</h1>"

@app.route("/predict", methods=['POST'])
def predict_acceptance():
    global setup_recourse
    
    try:
        json = request.get_json()
        param_dict = json['user_input']
    except:
        print('Error: invalid parameters')

    if setup_recourse.running():
        print('setup_recourse is still running')
        return 'Still computing the actions that can be made...'
    else:
        print('setup_recourse is done!')
        recourse_actions = setup_recourse.result()
        predicted_dict = recourse_actions.predict(param_dict)
        return predicted_dict

@app.route("/recourse", methods=['GET'])
def check_actions():
    global setup_recourse

    if setup_recourse.running():
        print('setup_recourse is still running')
        return 'Still computing the actions that can be made...'
    else:
        print('setup_recourse is done!')
        recourse_actions = setup_recourse.result()
        actions_table = recourse_actions.get_actions()
        return actions_table

@app.route("/person", methods=['GET'])
def check_person():
    global setup_recourse
    # return get_actions()
    if setup_recourse.running():
        print('setup_recourse is still running')
        return 'Still computing the actions that can be made...'
    else:
        print('setup_recourse is done!')
        recourse_actions = setup_recourse.result()
        person_table = recourse_actions.get_person()
        return person_table

if __name__ == "__main__":
    app.run(debug=True)
