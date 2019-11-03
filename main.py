"""
  main
"""

# pylint: disable=wildcard-import


if __name__ == "__main__":
  
  from hat.utils import *
  
  C = config()
  F = factory(C)

  F.train()
  F.val()

  # print(C.input_shape)
  # C.model.summary()
