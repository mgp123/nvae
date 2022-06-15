from typing import Any

from torch import nn

"""             
      +-------+   +-------+   +-------+       
      |       |   |       |   |       | optional      
--->---  m1   -->--  m2   -->--  m3   |------->
      |       |   |       |   |       |       
      +---|---+   +---|---+   +---|---+       
          |           |           |           
          |           |           |           
          V           V           V           
"""


class Splitter(nn.Sequential):

    def __init__(self, *args: Any, split_tail=False):
        """
        Splitter splits output for each module added. 
        After applying a module's transformation one branch is taken as input for the next module and the other one is taken as output.
        The architecture looks like a comb.
        Optionally, last module can also be split with split_tail.
        """
        super(Splitter,self).__init__(*args)
        self.split_tail = split_tail

    def forward(self, x):
        y = []
        last_output = x
        for module in self:
            last_output = module(last_output)
            y.append(last_output)
        if self.split_tail:
            y.append(last_output.clone())

        return y
