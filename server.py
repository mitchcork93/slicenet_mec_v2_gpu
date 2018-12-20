from autobahn.asyncio.websocket import WebSocketServerProtocol, WebSocketServerFactory
import trollius as asyncio
import uuid
import numpy
import cPickle
import logging
import sys
import json
import facial_recognition
import cv2
import opencv
from stroke_assessment import Assessment


class MECServerProtocol(WebSocketServerProtocol):

    web_connections = set()
    edge_connections = dict()
    obj_uuid = 0

    def __init__(self):
        WebSocketServerProtocol.__init__(self)
        self.obj_uuid = uuid.uuid4()

    def onConnect(self, request):
        print "Client connecting: {}".format(request.peer)

        if request.path == "/live":
            self.web_connections.add(self)

        elif request.path == "/get":
            print "new ambulance connection"

            ambulance_assessment = Assessment()
            self.edge_connections[self] = ambulance_assessment

    def onOpen(self):
        print"WebSocket connection open."

        #assessment = self.edge_connections[self]
        #self.sendMessage(json.dumps(assessment.start_test()), False)

    def onMessage(self, payload, isBinary):

        # get assessment object of connection

        try:
            assessment = self.edge_connections[self]
            client_message = json.loads(payload)

            if client_message["type"] == "init":
                self.sendMessage(json.dumps({"type": "get_image", "mode": client_message["mode"]}))

            elif client_message["type"] == "image":
                cv_image = numpy.asanyarray(cPickle.loads(str(client_message["payload"])))

                if client_message["mode"] == "diagnostic":
                    pain_image = opencv.detect_pain(cv_image)
                    self.sendMessage(json.dumps({"type": "receive_image", "mode": "diagnostic", "payload": cPickle.dumps(pain_image)}))

                if client_message["mode"] == "identify":

                    patient = facial_recognition.find_patient(cv_image)
                    
                    if patient.has_key("ERROR"):

                        if patient["ERROR"] == "FACES":
                            self.sendMessage(json.dumps({"ERROR": "Ensure only one person is in the image", "type": "get_image", "mode": "identify"}))
                        elif patient["ERROR"] == "PERSON":
                            self.sendMessage(json.dumps({"ERROR": "Patient not recognised", "type": "get_image", "mode": "identify"}))

                    elif patient["type"] == "patient":
                        self.sendMessage(json.dumps(patient))

                #self.sendMessage(json.dumps({"type": "get_image"}))

        except KeyError, e:
            print "no assessment found"

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def get_assessment_id(self):
        self.assessment_id = self.assessment_id + 1
        return self.assessment_id

    def sendDiagnostics(self, image):
        pass
        # convert data to image

    def check_result(self, assessment, result):

        current_stage = assessment.get_stage()

        if current_stage == "Complete":
            self.sendMessage(json.dumps({"command": "Complete"}), False)
            del self.edge_connections[self]
            assessment.send_results()
            self.sendClose()

        else:

            # if test is ready
            if assessment.has_timeout():
                assessment.calculate_result()
                self.sendMessage(json.dumps(assessment.start_next_stage()), False)

            else:
                assessment.save_result(result)
                assessment.increase_frame()

    def __hash__(self):
        return hash((self.obj_uuid))

    def __eq__(self, other):
        return (self.obj_uuid) == (other.obj_uuid)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)


if __name__ == '__main__':

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    root.addHandler(ch)

    factory = WebSocketServerFactory(u"ws://127.0.0.1:8888")
    factory.protocol = MECServerProtocol

    loop = asyncio.get_event_loop()
    coro = loop.create_server(factory, '127.0.0.1', 8888)
    server = loop.run_until_complete(coro)

    try:
        print 'Server Started...'
        loop.run_forever()
    except KeyboardInterrupt:
        print "Interrupt received... exiting"

    finally:
        server.close()
        loop.stop()
        loop.close()
