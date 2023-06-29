import tensorflow as tf


def conv_block(inputs, filters, activation, kernel_size=3, dilation_rate=1):
    cx = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)(
        inputs)
    cx = tf.keras.layers.BatchNormalization()(cx)
    if activation == 'relu':
        return tf.keras.layers.ReLU()(cx)
    elif activation == 'leaky_relu':
        return tf.keras.layers.LeakyReLU()(cx)
    elif activation is None:
        return cx
    else:
        print('Bad activation')
        sys.exit(0)


def multi_res_block(inputs, filters, activation, alpha=1.67):
    W = alpha * filters
    shortcut = conv_block(inputs, int(
        W * 0.167) + int(W * 0.333) + int(W * 0.5), kernel_size=1, activation=None)
    conv3x3 = conv_block(inputs, int(W * 0.167), activation=None)
    conv5x5 = conv_block(conv3x3, int(W * 0.333), activation=None)
    conv7x7 = conv_block(conv5x5, int(W * 0.5), activation=None)

    if activation is None:
        mresx = tf.keras.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
        mresx = tf.keras.layers.Add()([mresx, shortcut])
        return tf.keras.layers.BatchNormalization()(mresx)

    mresx = tf.keras.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
    mresx = tf.keras.layers.BatchNormalization()(mresx)
    mresx = tf.keras.layers.Add()([mresx, shortcut])
    if activation == 'leaky_relu':
        mresx = tf.keras.layers.LeakyReLU()(mresx)
    elif activation == 'relu':
        mresx = tf.keras.layers.ReLU()(mresx)
    else:
        print('Bad extivation')
        sys.exit(0)
    return tf.keras.layers.BatchNormalization()(mresx)


def dilated_multi_res_block(inputs, filters, activation, alpha=1.67, dilation_rate=2):
    W = alpha * filters
    shortcut = conv_block(inputs, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), kernel_size=1,
                          dilation_rate=dilation_rate, activation=None)
    conv3x3 = conv_block(inputs, int(W * 0.167),
                         dilation_rate=dilation_rate, activation=None)
    conv5x5 = conv_block(conv3x3, int(W * 0.333),
                         dilation_rate=dilation_rate, activation=None)
    conv7x7 = conv_block(conv5x5, int(W * 0.5),
                         dilation_rate=dilation_rate, activation=None)

    if activation is None:
        mresx = tf.keras.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
        mresx = tf.keras.layers.Add()([mresx, shortcut])
        return tf.keras.layers.BatchNormalization()(mresx)

    mresx = tf.keras.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
    mresx = tf.keras.layers.BatchNormalization()(mresx)
    mresx = tf.keras.layers.Add()([mresx, shortcut])
    if activation == 'leaky_relu':
        mresx = tf.keras.layers.LeakyReLU()(mresx)
    elif activation == 'relu':
        mresx = tf.keras.layers.ReLU()(mresx)
    else:
        print('Bad extivation')
        sys.exit(0)
    return tf.keras.layers.BatchNormalization()(mresx)


def res_path(inputs, filters, length, activation='relu'):
    shortcut = conv_block(inputs, filters, kernel_size=1, activation=None)
    rx = conv_block(inputs, filters, activation=activation)
    rx = tf.keras.layers.Add()([shortcut, rx])
    if activation == 'leaky_relu':
        rx = tf.keras.layers.LeakyReLU()(rx)
    elif activation == 'relu':
        rx = tf.keras.layers.ReLU()(rx)
    else:
        print('Bad extivation')
        sys.exit(0)

    rx = tf.keras.layers.BatchNormalization()(rx)

    for i in range(length - 1):
        shortcut = conv_block(rx, filters, kernel_size=1, activation=None)
        rx = conv_block(rx, filters, activation=activation)
        rx = tf.keras.layers.Add()([shortcut, rx])
        if activation == 'leaky_relu':
            rx = tf.keras.layers.LeakyReLU()(rx)
        elif activation == 'relu':
            rx = tf.keras.layers.ReLU()(rx)
        else:
            print('Bad extivation')
            sys.exit(0)
        rx = tf.keras.layers.BatchNormalization()(rx)

    return rx


def visual_attention_block(enc_input, dec_input, r=8):
    C = enc_input.shape[3]

    # Channel attention for decoder input
    f_ch_avg = tf.keras.layers.GlobalAvgPool2D()(dec_input)
    m_ch = MLP(f_ch_avg, C, r)
    # f_ch = tf.keras.layers.RepeatVector(dimension*dimension)(f_ch)
    # f_ch = tf.keras.layers.Reshape((dimension,dimension,C))(f_ch)

    # Spatial attention for decoder input
    f_sp_avg = tf.reduce_mean(dec_input, 3, keepdims=True)
    m_sp = tf.keras.layers.Conv2D(
        filters=1, kernel_size=3, padding='same', activation='sigmoid')(f_sp_avg)

    f_ch = tf.keras.layers.Multiply()([m_ch, enc_input])
    f_sp = tf.keras.layers.Multiply()([f_ch, m_sp])

    return f_sp


def MLP(inputs, C, r=8):
    mx = tf.keras.layers.Dense(C / r, activation='relu')(inputs)
    mx = tf.keras.layers.Dense(C, activation='relu')(mx)
    return mx


def dilated_multires_visual_attention(input_shape, starting_filters=16, with_dropout=False, activation='relu'):
    layer_filters = starting_filters
    # Encoder path
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = multi_res_block(model_input, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)
    sc_1 = res_path(x, layer_filters, 4)

    layer_filters *= 2
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)
    sc_2 = res_path(x, layer_filters, 3)

    layer_filters *= 2
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)
    sc_3 = res_path(x, layer_filters, 2)

    layer_filters *= 2
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = dilated_multi_res_block(x, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)
    sc_4 = res_path(x, layer_filters, 1)

    layer_filters *= 2
    x = dilated_multi_res_block(x, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)
    sc_5 = res_path(x, layer_filters, 1)

    # Bottleneck path
    layer_filters *= 2
    x = dilated_multi_res_block(x, layer_filters, activation=activation)
    if with_dropout:
        x = tf.keras.layers.Dropout(0.3)(x)

    # Decoder path
    layer_filters //= 2
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_5, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)

    layer_filters //= 2
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_4, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_3, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.2)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_2, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)

    layer_filters //= 2
    x = tf.keras.layers.Conv2DTranspose(
        filters=layer_filters, kernel_size=1, strides=2, padding='same')(x)
    x = multi_res_block(x, layer_filters, activation=activation)
    attn_output = visual_attention_block(sc_1, x)
    x = tf.keras.layers.Concatenate()([x, attn_output])
    if with_dropout:
        x = tf.keras.layers.Dropout(0.1)(x)

    # Output
    output = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model
