class Dataset(object):

  """
  这是一个数据集基类。

  你需要重写的方法是: 
  
    args()

  args里应当包含self._MISSION_LIST的定义，指定数据集可以执行的任务。
  """

  def __init__(self, *args, **kwargs):

    self.INPUT_SHAPE = ()
    self.NUM_CLASSES = 0

    self._list = ['mission', 'NUM_TRAIN', 'NUM_TEST', 'NUM_VAL', 'NUM_CLASSES', 'INPUT_SHAPE']
    self._dict = {}
    self._info_list = ['NUM_TRAIN', 'NUM_TEST', 'NUM_VAL', 'DATAINFO']
    self._info_dict = {}

    self._built = True
    self.mission = 'classfication'
    self.__dict__ = {**self.__dict__, **kwargs}
    self._MISSION_LIST = []
    self.DATAINFO = {}
    self.args()
    self._check_mission()
    self._built = False
  
  # built-in method

  def __setattr__(self, name, value):
    if '_built' not in self.__dict__:
      return super().__setattr__(name, value)
    if name == '_built':
      return super().__setattr__(name, value)
    if self._built:
      if name in self._list:
        self._dict[name] = value
      if name in self._info_list:
        self._info_dict[name] = value
    return super().__setattr__(name, value)

  # private method

  def _check_mission(self):
    if self.mission not in self._MISSION_LIST:
      raise Exception(f'Mission Error. The {type(self).__name__} does not have {self.mission} mission.') 
    elif self.mission == 'classfication':
      self.DATAINFO = {'INPUT_SHAPE': self.INPUT_SHAPE, 'NUM_CLASSES': self.NUM_CLASSES}
  
  def args(self):
    raise NotImplementedError

  # public method

  def ginfo(self):
    return self._info_dict, self._dict
    
if __name__ == "__main__":
  a = Dataset()
  print(a.mission)
  b = Dataset(mission='object_detection')
  print(b.mission)
  