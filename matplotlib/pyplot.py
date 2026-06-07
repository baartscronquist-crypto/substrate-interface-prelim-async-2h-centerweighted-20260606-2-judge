class _DummyAxes:
    def contourf(self, *args, **kwargs):
        return None

    def plot(self, *args, **kwargs):
        return []

    def fill_between(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def set_xscale(self, *args, **kwargs):
        return None


class _AxesGrid:
    def __init__(self, rows, cols):
        self._axes = [[_DummyAxes() for _ in range(cols)] for _ in range(rows)]

    def ravel(self):
        return [ax for row in self._axes for ax in row]

    def __iter__(self):
        return iter(self._axes)


class _DummyFigure:
    def colorbar(self, *args, **kwargs):
        return None

    def suptitle(self, *args, **kwargs):
        return None

    def savefig(self, *args, **kwargs):
        return None


def subplots(rows=1, cols=1, *args, **kwargs):
    fig = _DummyFigure()
    if rows == 1 and cols == 1:
        return fig, _DummyAxes()
    if rows == 1:
        return fig, [_DummyAxes() for _ in range(cols)]
    return fig, _AxesGrid(rows, cols)


def close(*args, **kwargs):
    return None
