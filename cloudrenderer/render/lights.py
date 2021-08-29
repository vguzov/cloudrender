import numpy as np

class Light:
    def __init__(self, light_type):
        self.type = light_type

    @staticmethod
    def normalize(v):
        vnorm = np.linalg.norm(v)
        if vnorm > 0:
            return v/vnorm
        else:
            return v


class DirectionalLight(Light):
    def __init__(self, direction: np.ndarray, intensity: np.ndarray):
        super().__init__('directional')
        self.direction = self.normalize(direction)
        self.intensity = intensity
