class String1:
    def __init__(self, string):
        self.string = string

    def __sub__(self, other):
        # When doing a - b
        if isinstance(other, String1):
            return String1(self.string + " - " + other.string)
        return String1(self.string  + " - " + other)
    
    def __rsub__(self, other):
    # When doing string - object
        return String1(other  + " - " + self.string)
    
    def __mul__(self, other):
        # When doing a * b
        if isinstance(other, String1):
            return String1(self.string + " * " + other.string)
        return String1(self.string  + " * " + other)
    
    def __rmul__(self, other):
    # When doing string - object
        return String1(other  + " * " + self.string)
    
    def __repr__(self):
        return f"{self.string}"