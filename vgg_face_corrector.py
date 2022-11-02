def fix_vgg():
    filename = "/home/jarred/anaconda3/envs/Orfao_Masters/lib/python3.8/site-packages/keras_vggface/models.py"
    # filename = "/home/jarred/Documents/virtual_envs/Orfao_Masters/lib/python3.8/site-packages/keras_vggface/models.py"
    text = open(filename).read()
    open(filename, "w+").write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

if __name__ == "__main__":
    fix_vgg()