from enum import Enum

class Phase(Enum):
    train = 1
    test = 2
    validation = 3

class Base:
    def __init__(self):
        self.phase = Phase.test