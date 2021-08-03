import numpy as np

class Light:
    def __init__(self, light_type):
        self.type = light_type

class DirectionalLight(Light):
    def __init__(self, direction: np.ndarray, intensity: np.ndarray):
        super().__init__('directional')
        self.direction = direction
        self.intensity = intensity
