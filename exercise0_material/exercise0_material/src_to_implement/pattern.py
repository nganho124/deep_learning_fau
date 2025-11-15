import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution=10, tile_size=1):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        pattern = np.array([[0, 1], [1, 0]])
        num_tiles = self.resolution // self.tile_size

        reps = (num_tiles // 2, num_tiles // 2)
        tiled_pattern = np.tile(pattern, reps)

        self.output = np.kron(tiled_pattern, np.ones((self.tile_size, self.tile_size)))

        self.output = self.output[:self.resolution, :self.resolution]
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray', interpolation='nearest')
        plt.show()

class Circle:
    
    def __init__(self, resolution=10, radius=1, position=(0,0)):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None
        
    def draw(self):
        x = np.linspace(0, self.resolution - 1, self.resolution)
        y = np.linspace(0, self.resolution - 1, self.resolution)
        X, Y = np.meshgrid(x, y)
        F = (X - self.position[0])**2 + (Y - self.position[1])**2
        self.output = (F <= self.radius**2)
        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output, cmap='gray', interpolation='nearest')
        plt.show()
        
class Spectrum:
    
    def __init__(self, resolution=10):
        self.resolution = resolution
        
    def draw(self):
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        X, Y = np.meshgrid(x, y)
        R = X
        G = Y
        B = 1 - X
        
        imp = np.dstack((R, G, B))
        self.output = np.clip(imp, 0, 1)
        
        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output)
        plt.axis('off')
        plt.show()