from flask import Flask,jsonify,request

app = Flask(__name__)


## initialize the data 

items = [
    {'id':1,'name':"item1",'des':"This is item1"},
    {'id':2,'name':"item2",'des':"This is item2"}
]


@app.route('/')
def home():
    return "Welcome to the sample To Do List App"


@app.route('/items',methods = ['GET'])
def get_items():
    return jsonify(items)


## get : Get the specific element by ID
@app.route('/items/<int:item_id>',methods = ['GET'])
def get_item(item_id):
    item = next((item for item in items if item['id'] == item_id),None)
    if item is not None:
        return jsonify(item)
    return jsonify({"error":"The item is not found...!"})


## Add new item using the post method 
@app.route('/items',methods = ['POST'])
def create_item():
    if not request.json or not 'name' in request.json:
        return jsonify({"Error : " : "item not found"})
    new_item = {
        'id' : items[-1]['id'] + 1 if items else 1,
        'name': request.json['name'],
        "description" : request.json["description"]
    }
    items.append(new_item)
    return jsonify(new_item)

## Put : Update an exisiting item

@app.route('/items/<int:item_id>',methods = ['PUT'])
def update_item(item_id):
    item = next((item for item in items if item['id'] == item_id),None)
    if item is not None:
        return jsonify(item)
    return jsonify({"error":"The item is not found...!"})

@app.route('/items/<int:item_id>',methods = ['DELETE'])
def delete_item(item_id):
    global items 
    item = next((item for item in items if item['id'] != item_id),None)
    return jsonify({"result":"Item Deleted...!"})

if __name__ == '__main__':
    app.run(debug=True)