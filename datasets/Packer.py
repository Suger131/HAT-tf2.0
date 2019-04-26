class Packer:

  def __init__(self):
    pass
  
  def ginfo(self):
    return {i: self.__dict__[i] for i in ['NUM_TRAIN', 'NUM_TEST', 'NUM_CLASSES', 'IMAGE_SHAPE']}
    