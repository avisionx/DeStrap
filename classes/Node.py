import string
import random

TEXT_PLACE_HOLDER = "[]"

class Node:

    def __init__(self, key, parent_node, content_holder):
        self.content_holder = content_holder
        self.setParentKey(parent_node, key)
        self.initializeChilds()

    def setParentKey(self, parent_node, key):
        self.key = key
        self.parent = parent_node

    def initializeChilds(self):
        self.children = []

    def rendering_function(self, key, value):
        if(self.indexOf(key, "btn") != -1):
            value = value.replace(TEXT_PLACE_HOLDER, self.get_random_text())
        elif(self.indexOf(key, "text") != -1):
            value = value.replace(TEXT_PLACE_HOLDER, self.get_random_text(length_text=56, space_number=7, with_upper_case=False))
        elif(self.indexOf(key, "title") != -1):
            value = value.replace(TEXT_PLACE_HOLDER, self.get_random_text(length_text=5, space_number=0))
        return value
    
    def indexOf(self, key, text):
        return key.find(text)

    def add_child(self, child):
        if(self.children is None):
            self.children = []
        self.children.append(child)

    def render(self, mapping, rendering_function=None):

        content = None
        
        for child in self.children:
            placeholder = child.render(mapping, self.rendering_function)
            if(placeholder is not None):
                if(content is None):
                    content = ""
                content += placeholder
            else:
                self = None
                return

        value = mapping.get(self.key, None)

        if(value is not None):
            
            if(rendering_function is not None):
                value = self.rendering_function(self.key, value)

        else:
            self = None
            return None
        
        if(len(self.children) != 0):
            value = value.replace(self.content_holder, content)

        return value

    def get_random_text(self, space_number=1, length_text=10, with_upper_case=True):
        
        results = []
        
        while(length_text >= len(results)):
            char = random.choice(string.ascii_letters[:26])
            results.append(char)

        current_spaces = []
        
        if(with_upper_case):
            results[0] = results[0].upper()

        while(space_number > len(current_spaces)):

            space_pos = random.randint(2, length_text - 3)
            
            if space_pos in current_spaces:
                break
            else:
                results[space_pos] = " "
            
            current_spaces.append(space_pos)

            if(with_upper_case):
                last_char = results[space_pos - 1].upper()
                results[space_pos + 1] = last_char

        return ''.join(results)