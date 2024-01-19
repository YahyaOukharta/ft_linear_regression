import os
from ast import literal_eval

class ThetaCache:
    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(self.filename):
            open(self.filename, 'w').close()

    def set(self, key, value):
        with open(self.filename, 'a') as file:
            file.write(f"{key}={value}\n")

    def get(self, key):
        with open(self.filename, 'r') as file:
            for line in file:
                k, _, v = line.partition('=')
                if k == str(key):
                    return literal_eval(v.strip())
        return 0, 0


theta_cache = ThetaCache(".cache")