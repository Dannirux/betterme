from flask import Flask, json, jsonify, request
from test import chatbot_response
import ast
application = Flask(__name__)

@application.route("/")
def hello_world():
    return "<p>Hello, World! this is chatbot api</p>"


@application.route('/', methods=['GET', 'POST'])
def returnChatbotResponse():
    if request.method == 'POST':
        userQuery = (request.args.get('userQuery'))
        result = {
            "user_query": userQuery,
            "chatbot_response": chatbot_response(userQuery.lower())
        }
        return jsonify(result)
    else:
        return "<p>Please use proper API POST request call for chatbot response</p>"


"""
if __name__ == '__main__':
    application.run(host="0.0.0.0", port=5000, debug=True)
 """
