"""
  hat.utils.Counter
"""


__all__ = [
  'Counter'
]


class Counter(object):
  """
    Counter 
    
    to count and remember the number.

    Argument:
      name: `str`, the count number's name.

    Return:
      A `str` number.

    Usage:
      The `Counter` can be used as a `str` object.
      Or like a function. 
      This two method use the `default initial value 1`.
      And each visit will automatically increase by 1.
      Or use the `get/set` method. 
      The `set` method can use a free initial value.
      The `get` method do not change the value.

    Sample:
    ```python
      x = f"{Counter('a')}"
      print(Counter('a'))
      Counter()('a')

      a=Counter()
      a.set('b', 5)
      a.get('b')
    ```
  """
  count = {}

  def __init__(self, name=None, *args, **kwargs):
    self.name = name

  def __str__(self):
    if self.name:
      if self.name not in Counter.count:
        Counter.count[self.name] = 1
      else:
        Counter.count[self.name] += 1
      return str(Counter.count[self.name])
    return None
  
  def __call__(self, name):
    if name:
      if name not in Counter.count:
        Counter.count[name] = 1
      else:
        Counter.count[name] += 1
      return str(Counter.count[name])
    return None

  def get(self, name):
    """
      Get the count number but not change.

      Argument:\n
        name: `str`, the count number's name.
    """
    if name in Counter.count:
      return str(Counter.count[name])
    return None
      
  def set(self, name, value):
    """
      Set the count number.

      Argument:\n
        name: `str`, the count number's name.\n
        value : `int`, the count number's value.
    """
    if type(value) != int:
      raise Exception(f"Value must be a int, but got {type(value).__name__}")
    Counter.count[name] = value


if __name__ == "__main__":
  a = Counter()
  print(a.get('a'))
  a.set('b', 5)
  print(a.get('b'))
  print(Counter('b'))
  print(a('b'))
  print(Counter()('b'))
  print(a.get('b'))
  a.set('c', 'b')
