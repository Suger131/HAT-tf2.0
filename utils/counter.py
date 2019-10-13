"""
  hat.utils.counter
"""


__all__ = [
  'counter'
]


class counter(object):
  """
    counter 
    
    to count and remember the number.

    Argument:
      name: `str`, the count number's name.

    Return:
      A `str` number.

    Usage:
      The `counter` can be used as a `str` object.
      Or like a function. 
      This two method use the `default initial value 1`.
      And each visit will automatically increase by 1.
      Or use the `get/set` method. 
      The `set` method can use a free initial value.
      The `get` method do not change the value.

    Sample:
    ```python
      x = f"{counter('a')}"
      print(counter('a'))
      counter()('a')

      a=counter()
      a.set('b', 5)
      a.get('b')
    ```
  """
  count = {}

  def __init__(self, name=None, *args, **kwargs):
    self.name = name

  def __str__(self):
    if self.name:
      if self.name not in counter.count:
        counter.count[self.name] = 1
      else:
        counter.count[self.name] += 1
      return str(counter.count[self.name])
    return None
  
  def __call__(self, name):
    if name:
      if name not in counter.count:
        counter.count[name] = 1
      else:
        counter.count[name] += 1
      return str(counter.count[name])
    return None

  def get(self, name):
    """
      Get the count number but not change.

      Argument:\n
        name: `str`, the count number's name.
    """
    if name in counter.count:
      return str(counter.count[name])
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
    counter.count[name] = value


if __name__ == "__main__":
  a = counter()
  print(a.get('a'))
  a.set('b', 5)
  print(a.get('b'))
  print(counter('b'))
  print(a('b'))
  print(counter()('b'))
  print(a.get('b'))
  a.set('c', 'b')
