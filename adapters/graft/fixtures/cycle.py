def alpha():
    return beta()

def beta():
    return alpha()

class Example:
    def method(self):
        return alpha()
