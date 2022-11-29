from json import JSONDecoder, JSONEncoder
import cv2 as cv

class Feature:
    def __init__(self, kp, desc, frame : cv.UMat = None) -> None:
        self.frame = frame
        self.keypoints = kp
        self.descriptors = desc

class Track:
    def __init__(self) -> None:
        self.descriptors = []

class FeatureJSONEncoder(JSONEncoder):
    def default(self, obj : Feature):
        return {
            'kp' : [{
                    'angle': k.angle, 
                    'class_id' : k.class_id,
                    'octave' : k.octave,
                    'pt' : k.pt,
                    'response': k.response,
                    'size' : k.size
                } for k in obj.keypoints],
                'desc' : [x.tolist() for x in obj.descriptors]
            }

class FeatureJSONDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs):
        JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj, dict):
            if 'angle' in obj:
                return cv.KeyPoint(
                    int(obj.get('pt')[0]), 
                    int(obj.get('pt')[1]),
                    float(obj.get('size')),
                    angle=float(obj.get('angle')),
                    response=float(obj.get('response')),
                    octave=int(obj.get('octave')),
                    class_id=int(obj.get('class_id')))
            if 'kp' in obj:
                return Feature(**obj)

        if isinstance(obj, list):
            for i in range(0, len(obj)):
                obj[i] = self.object_hook(i)
            return obj

        return obj