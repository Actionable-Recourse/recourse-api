from flask import Flask
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from recourse_api import setup

# DOCS https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
executor = ThreadPoolExecutor(2)
app = Flask(__name__)
setup_recourse = executor.submit(setup)

@app.route("/", methods=['GET'])
def index():
    return "<h1>This is Recourse API!</h1>"

@app.route("/recourse", methods=['GET'])
def check_actions():
    global setup_recourse
    # return get_actions()
    if setup_recourse.running():
        print('setup_recourse is still running')
        return 'Still computing the actions that can be made...'
    else:
        print('setup_recourse is done!')
        recourse_actions = setup_recourse.result()
        actions_table = recourse_actions.get_actions()
        return actions_table
    

if __name__ == "__main__":
    app.run(debug=True)
