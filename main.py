import tensorflow as tf
from Arg import ARGS


if __name__ == "__main__":

  ARGS.gimage()

  ARGS.train()

  ARGS.test()

  ARGS.save()