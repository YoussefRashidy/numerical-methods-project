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
    # When doing string * object
        return String1(other  + " * " + self.string)
    
    def __truediv__(self, other):
        # When doing a / b
        if isinstance(other, String1):
            return String1(self.string + " / " + other.string)
        return String1(self.string  + " / " + other)
    
    def __rtruediv__(self, other):
    # When doing string / object
        return String1(other  + " / " + self.string)
    
    def __add__(self, other):
        # When doing a + b
        if isinstance(other, String1):
            return String1(self.string + " + " + other.string)
        return String1(self.string  + " + " + other)
    
    def __radd__(self, other):
    # When doing string + object
        return String1(other  + " + " + self.string)
    
    def __neg__(self):
        # When doing -a
        return String1( " - " + self.string)
    
    
    def __floordiv__(self, other):
        # When doing a // b
        if isinstance(other, String1):
            return String1(self.string + " // " + other.string)
        return String1(self.string  + " // " + other)
    
    def __rfloordiv__(self, other):
    # When doing string // object
        return String1(other  + " // " + self.string)
    
    def __repr__(self):
        return f"{self.string}"