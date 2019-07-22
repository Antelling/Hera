import os, json
files = os.listdir("detailed_penalty_model_info")

keys = []

for file in files:
    data = json.loads(open(os.path.join("detailed_penalty_model_info", file)).read())
    data = data[data["name"]]

