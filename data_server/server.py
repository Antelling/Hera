from http.server import BaseHTTPRequestHandler, HTTPServer
import time

import os

import data

file_path = os.path.dirname(os.path.realpath(__file__))

hostName = "localhost"
hostPort = 3000

index = open(os.path.join(file_path, "index.html")).read()
index = bytes(index, "utf-8")


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(index)

    def do_POST(self):
        # Doesn't do anything with posted data
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        post_data = str(post_data)[2:].split("&")
        person_obj = {}
        for header in post_data:
            header = header.split("=")
            person_obj[header[0]] = header[1]
        person_obj["position"] = [person_obj["ext"],
                                  person_obj["emo"],
                                  person_obj["agr"],
                                  person_obj["con"],
                                  person_obj["tel"]]
        del person_obj["ext"]
        del person_obj["emo"]
        del person_obj["agr"]
        del person_obj["con"]
        del person_obj["tel"]
        person_obj["display"] = True
        person_obj["grade"] = int(person_obj["grade"])
        data.set.save_person(**person_obj)
        self.wfile.write(bytes("saved", "utf-8"))


myServer = HTTPServer((hostName, hostPort), MyServer)
print(time.asctime(), "Server Starts - %s:%s" % (hostName, hostPort))

try:
    myServer.serve_forever()
except KeyboardInterrupt:
    pass

myServer.server_close()
print(time.asctime(), "Server Stops - %s:%s" % (hostName, hostPort))
