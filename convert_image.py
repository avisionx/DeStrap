#!/usr/bin/env python
from classes.Runner import *

if __name__ == "__main__":
    
    input_path = "./input.png"
    
    print_generated_output = True
    original_gui = "./input.gui"
    
    get_accuracy = True
    get_html = True

    runner = Runner()
    runner.convert_image(input_path, original_gui, print_generated_output, get_accuracy, get_html)