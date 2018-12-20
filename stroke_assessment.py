import json
import operator


class Assessment:

    #frame_limit = 200
    #current_action = ""
    #frame_count = 0
    #results = dict()

    def __init__(self):

        self.results = dict()
        self.current_stage = 1
        self.frame_count = 0
        self.frame_limit = 10
        self.test_expectations = {1: "Patient was capable of spreading fingers", 2: "Patient was capable of spreading fingers", 3: "Complete"}
        self.result_tracker = {"SPREAD": 0, "PARTIAL:": 0, "NONE": 0}
        self.result_text = {"SPREAD": "Patient was capable of spreading fingers", "PARTIAL": "Patient could partially spread fingers", "NONE": "Patient could not spread fingers"}

    def increase_frame(self):
        self.frame_count = self.frame_count + 1

    def reset_frame(self):
        self.frame_count = 0

    def start_test(self):
        return {"command": self.get_instructions(self.current_stage)}

    def save_result(self, result):
        self.result_tracker[result] = self.result_tracker[result] + 1

    def calculate_result(self):

        analysis = max(self.result_tracker.iteritems(), key=operator.itemgetter(1))[0]
        self.results[self.current_stage] = self.result_text[analysis]

    def start_next_stage(self):
        self.current_stage = self.current_stage + 1
        self.reset_frame()
        return {"command": self.get_instructions(self.current_stage)}

    def get_stage(self):
        return self.test_expectations[self.current_stage]

    def get_instructions(self, stage):

        try:
            instructions = {
                1: "Hold your right hand to the camera and spread your fingers",
                2: "Hold your left hand to the camera and spread your fingers",
                3: "Complete"
            }

            return instructions[stage]

        except KeyError:
            print 'No Instruction For Stage: {}'.format(stage)

    def has_timeout(self):

        if self.frame_count < self.frame_limit:
            return False
        else:
            return True

    def send_results(self):

        print json.dumps(self.results, indent=4, sort_keys=True)

        """
            Method sends captured landmarks (in JSON format) to MAT C++ server for processing and predicting emotion.
             ****************************** METHOD NOT CURRENTLY IN USE *********************************************
            :param landmarks:
            :return emotion:
        

        # Local MAT server
        url = 'http://localhost:9000/test'
        headers = {'content-type': 'application/json'}

        response = requests.post(url, data=json.dumps(landmarks), headers=headers)
        #return response.json()
        
        """