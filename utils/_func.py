def _writer(func):
  def _inner(self, *args, **kwargs):
    with open(self._log_dir, 'a+') as f:
      for i in args: f.write(i + '\n')
    func(self, *args, **kwargs)
  return _inner