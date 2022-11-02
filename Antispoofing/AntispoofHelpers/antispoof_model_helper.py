import math


def initialise_tf():
    import tensorflow as tf
    try:
        # fix memory issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

def create_vit_b32(image_size=224, num_classes=2, verbose=False, include_traditional=False, use_hsv=False):
    initialise_tf()
    import tensorflow as tf
    from vit_keras import vit

    def to_hsv(x):
        return tf.image.rgb_to_hsv(x)

    vit_model = vit.vit_b32(
        image_size=image_size,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=num_classes
    )

    vit_model.trainable = False

    degrees = 15
    percentage = (degrees * (math.pi / 180.0)) / (2 * math.pi)
    if include_traditional:
        model = tf.keras.Sequential([
            # data aug
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(percentage),
            tf.keras.layers.RandomZoom((-0.2, 0)),

            tf.keras.layers.Lambda(vit.preprocess_inputs),
            vit_model,
            tf.keras.layers.Dense(num_classes, 'softmax')
        ],
            name='vision_transformer')
    elif use_hsv:
        model = tf.keras.Sequential([
            tf.keras.layers.Lambda(to_hsv),
            tf.keras.layers.Lambda(vit.preprocess_inputs),
            vit_model,
            tf.keras.layers.Dense(num_classes, 'softmax')
        ],
            name='vision_transformer')
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Lambda(vit.preprocess_inputs),
            vit_model,
            tf.keras.layers.Dense(num_classes, 'softmax')
        ],
            name='vision_transformer')
    if verbose:
        model.summary()
    return model

if __name__ == '__main__':
    create_vit_b32()