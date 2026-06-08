class Normalize:
    def __init__(self, vmin=None, vmax=None, clip=False):
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip

    def __call__(self, value):
        return value
