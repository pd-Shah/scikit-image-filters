class WindowShape(Exception):
    '''
    check window shape
    '''
    def __init__(self, message):
        self.message=message
    def __str__(self):
        return repr(self.message)


class KernelShapeError(WindowShape):
    '''
    check kernel shape in ["square", "disk", "rectangle"]
    '''
    pass
