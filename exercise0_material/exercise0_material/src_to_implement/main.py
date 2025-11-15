from exercise0_material.src_to_implement.pattern import Checker, Circle, Spectrum
from exercise0_material.src_to_implement.generator import ImageGenerator

if __name__ == "__main__":
    # Test Checker pattern
    checker = Checker(resolution=10, tile_size=10)
    checker.draw()
    checker.show()
    
    # Test Circle pattern
    circle = Circle(resolution=100, radius=30, position=(50, 50))
    circle.draw()
    circle.show()
    
    # Test Spectrum pattern
    spectrum = Spectrum(resolution=256)
    spectrum.draw()
    
    # Test ImageGenerator
    label_fic = ImageGenerator(file_path="exercise_data/", 
                           label_path="Labels.json", 
                           batch_size=16, 
                           image_size=(32,32), 
                           rotation=True, 
                           mirroring=False, 
                           shuffle=True)
    label_fic.show()