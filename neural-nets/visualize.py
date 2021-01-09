import time

import numpy as np
import torchvision as tv
import visdom


class Visualizer:
    """
    Capsulating visdom, still can use self.vis.function`
    """

    def __init__(self, env="default", **kwargs):
        import visdom

        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ""

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(
            Y=np.array([y]),
            X=np.array([x]),
            win=name,
            opts=dict(title=name),
            update=None if x == 0 else "append",
        )
        self.index[name] = x + 1

    def __getattr__(self, name):
        return getattr(self.vis, name)
