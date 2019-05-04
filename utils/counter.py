"""
  计数器类

"""


class Counter(object):

  count = {}

  def __init__(self, name, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.name = name

  def __str__(self):
    if self.name not in Counter.count:
      Counter.count[self.name] = 1
    else:
      Counter.count[self.name] += 1
    return str(Counter.count[self.name])
    