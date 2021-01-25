
class Config(object):
    BATCH_SIZE_LIST = [2**n for n in range(6, 8)]
    ADAM_LEARNING_RATE = 1E-6  # 1E-6 in AdamBandit paper
    SAVE_PLOT_DIR = './'  #  r'C:/Users/ivbarrie'
    EPOCHS=10
    