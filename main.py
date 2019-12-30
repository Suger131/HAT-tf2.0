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
  import hat
  C = hat.Config()
  F = hat.Factory(C)
  F.run()

