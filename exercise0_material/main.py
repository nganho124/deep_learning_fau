from exercise0_material.src_to_implement.pattern import Checker, Circle

if __name__ == "__main__":
    # Test Checker pattern
    checker = Checker(resolution=10, tile_size=10)
    checker.draw()
    checker.show()