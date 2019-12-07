"""
  hat.main

  程序入口
  通过Config配置参数
  通过Factory进行训练、测试等操作
"""


if __name__ == "__main__":
  
  import hat
  
  C = hat.Config()
  F = hat.Factory(C)
  F.train()
  F.val()

  # print(C.input_shape)
  # C.model.summary()
