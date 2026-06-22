import tensorflow as tf
import numpy as np
import cv2


def make_gradcam_heatmap(img_array, model):

    base_model = model.get_layer("mobilenetv2_1.00_224")

    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer("out_relu").output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        x = model.layers[0](img_array)

        conv_outputs, features = conv_model(x)

        tape.watch(conv_outputs)

        x = model.layers[2](features)
        x = model.layers[3](x)
        predictions = model.layers[4](x)

        if predictions[0][0] < 0.5:
            loss = 1 - predictions[:, 0]
        else:
            loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def generate_gradcam_image(img_array, original_img, model):

    heatmap = make_gradcam_heatmap(
        img_array,
        model
    )

    original_img = np.array(original_img)

    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    gradcam_img = cv2.addWeighted(
        original_img,
        0.6,
        heatmap,
        0.4,
        0
    )

    return gradcam_img