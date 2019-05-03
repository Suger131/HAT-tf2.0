class Dataset(object):

  """
  这是一个数据集基类。

  你需要重写的方法是: 
  
    args()

  args里应当包含self._MISSION_LIST的定义，指定数据集可以执行的任务。
  """

  def __init__(self, *args, **kwargs):
    self.mission = 'classfication'
    self._MISSION_LIST = []
    self.DATAINFO = {}
    self.__dict__ = {**self.__dict__, **kwargs}
    self.args()
    self._check_mission()
  
  # private method

  def _check_mission(self):
    if self.mission not in self._MISSION_LIST:
      raise Exception(f'Mission Error. The {type(self).__name__} does not have {self.mission} mission.') 
    elif self.mission == 'classfication':
      self.DATAINFO = {'IMAGE_SHAPE': self.IMAGE_SHAPE, 'NUM_CLASSES': self.NUM_CLASSES}
  
  def args(self):
    raise NotImplementedError

  # public method

  def ginfo(self):
    return {i: self.__dict__[i] for i in ['NUM_TRAIN', 'NUM_TEST', 'DATAINFO']}
    
if __name__ == "__main__":
  a = Dataset()
  print(a.mission)
  b = Dataset(mission='object_detection')
  print(b.mission)
  