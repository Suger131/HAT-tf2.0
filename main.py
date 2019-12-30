# -*- coding: utf-8 -*-
"""Main

  File:
    /hat/main

  Description:
    The entry of the program. 
    Use Config for configuring parameters.
    Use Factor for training, testing, and more.
"""


if __name__ == "__main__":
  from hat import core
  C = core.Config()
  F = core.Factory(C)
  F.run()

