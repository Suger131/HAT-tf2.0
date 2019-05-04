"""
  计数器类

"""


class Counter(object):

  count = {}

  def __init__(self, *args, **kwargs):
    return super().__init__(*args, **kwargs)

  def get(self, name):
    if name not in Counter.count:
      Counter.count[name] = 1
    else:
      Counter.count[name] += 1
    return Counter.count[name]