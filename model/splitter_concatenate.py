from typing import Any

from torch import nn

"""             
+-------+     +-------+     +-------+       +-------+     +-------+     +-------+
|       |     |       |     |       |       |       |     |       |     |       |
|       ------>       ------>       |   +   |       ------>       ------>       |
|       |     |       |     |       |       |       |     |       |     |       |
+---|---+     +---|---+     +---|---+       +---|---+     +---|---+     +---|---+
    |             |             |               |             |             |    
    v             v             v               v             v             v   

  = 

+-------+     +-------+     +-------+      +-------+     +-------+     +-------+
|       |     |       |     |       |      |       |     |       |     |       |
|       ------>       ------>       ------->       ------>       ------>       |
|       |     |       |     |       |      |       |     |       |     |       |
+---|---+     +---|---+     +---|---+      +---|---+     +---|---+     +---|---+
    |             |             |              |             |             |    
    v             v             v              v             v             v    

"""


class SplitterConcatenate(nn.Sequential):

    def __init__(self, *args):
        """
        Concatenates multiple splitters into a bigger one
        """
        super(SplitterConcatenate, self).__init__(*args)

    # note that we can't check if module types are Splitter
    # because they can be hidden inside another module

    def forward(self, x):
        y = []
        last_output = x
        for module in self:
            l = module(last_output)
            last_output = l[-1]
            y += l
        return y
