class Packer:

  def __init__(self):
    pass
  
  def ginfo(self, Args):
    for i in ['NUM_TRAIN', 'NUM_TEST', 'NUM_CLASSES', 'IMAGE_SIZE', 'IMAGE_DEPTH']:
      Args.__dict__[i] = self.__dict__[i]
