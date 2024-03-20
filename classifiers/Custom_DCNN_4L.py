import time
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling1D, Input,SeparableConv1D,DepthwiseConv1D, Flatten,Dense,Reshape, BatchNormalization, ELU, Permute, MaxPooling1D
from classifiers.classifiers import predict_model
from utils.classifier_tools import create_class_weight
from utils.tools import save_logs


class Classifier_Disjoint_CNN:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        if verbose:
            print('Creating Disjoint_CNN Classifier')
        self.verbose = verbose
        self.output_directory = output_directory
        # Build Model -----------------------------------------------------------
        self.model = self.build_model(input_shape, nb_classes)
        # -----------------------------------------------------------------------
        if verbose:
            self.model.summary()
        # self.model.save_weights(self.output_directory + 'model_init.weights.h5')

    def build_model(self, input_shape, nb_classes):
        X_input = Input(shape=input_shape[:2])
        # Reshape input to make it suitable for 1D convolutions
        X_reshaped = Permute((2, 1))(X_input)  # Swap dimensions to convert from (batch_size, height, width) to (batch_size, width, height)
        X_reshaped = Reshape((input_shape[1], input_shape[0]))(X_reshaped)  # Reshape to (batch_size, width, height)

        # Temporal Convolutions
        conv1 = DepthwiseConv1D(kernel_size=8, strides=1, padding='same', name="depth")(X_reshaped)
        conv1 = BatchNormalization()(conv1)
        conv1 = ELU(alpha=1.0)(conv1)
        # Spatial Convolutions
        conv1 = SeparableConv1D(64, kernel_size=input_shape[1], strides=1, padding='same', depthwise_initializer='glorot_uniform', name="sept")(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = ELU(alpha=1.0)(conv1)

        # Temporal Convolutions
        conv2 = DepthwiseConv1D(kernel_size=5, strides=1, padding='same', name="depth2")(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = ELU(alpha=1.0)(conv2)
        # Spatial Convolutions
        conv2 = SeparableConv1D(64, kernel_size=conv2.shape[2], strides=1, padding='same', depthwise_initializer='glorot_uniform', name="sept2")(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = ELU(alpha=1.0)(conv2)

        # Temporal Convolutions
        conv3 = DepthwiseConv1D(kernel_size=5, strides=1, padding='same', name="depth3")(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = ELU(alpha=1.0)(conv3)
        # Spatial Convolutions
        conv3 = SeparableConv1D(64, kernel_size=conv3.shape[2], strides=1, padding='same', depthwise_initializer='glorot_uniform', name="sept3")(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = ELU(alpha=1.0)(conv3)

        # Temporal Convolutions
        conv4 = DepthwiseConv1D(kernel_size=5, strides=1, padding='same', name="depth4")(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = ELU(alpha=1.0)(conv4)
        # Spatial Convolutions
        conv4 = SeparableConv1D(64, kernel_size=conv4.shape[2], strides=1, padding='same', depthwise_initializer='glorot_uniform', name="sept4")(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = ELU(alpha=1.0)(conv4)

        # Pooling layers
        max_pool = MaxPooling1D(pool_size=5, strides=None, padding='valid')(conv4)
        gap = GlobalAveragePooling1D()(max_pool)

        # Flatten layer
        flatten = Flatten()(gap)

        # Dense layers
        output_layer = Dense(nb_classes, activation='softmax')(flatten)

        # Create model
        model = keras.models.Model(inputs=X_input, outputs=output_layer)

        return model

    def fit(self, Ximg_train, yimg_train, Ximg_val, yimg_val, epochs, batch_size):
        if self.verbose:
            print('[Disjoint_CNN] Training Custom_Disjoint_CNN Classifier')

        mini_batch_size = int(min(Ximg_train.shape[0] / 10, batch_size))
        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
        file_path = self.output_directory + 'best_model.keras'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        # create class weights based on the y label proportions for each class
        class_weight = create_class_weight(yimg_train)
        start_time = time.time()
        self.hist = self.model.fit(Ximg_train, yimg_train,
                                   validation_data=[Ximg_val, yimg_val],
                                   class_weight=class_weight,
                                   verbose=0,
                                   epochs=epochs,
                                   batch_size=mini_batch_size,
                                   callbacks=self.callbacks)
        self.duration = time.time() - start_time

        keras.models.save_model(self.model, self.output_directory + 'model.keras')
        print('[Disjoint_CNN] Training done!, took {}s'.format(self.duration))

    def predict(self, X_img, y_img, best):
        if best:
            print(self.output_directory)
            model = keras.models.load_model(self.output_directory + 'best_model.keras')
        else:
            model = keras.models.load_model(self.output_directory + 'model.keras')
        model_metrics, conf_mat, y_true, y_pred = predict_model(model, X_img, y_img, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)
        keras.backend.clear_session()
        return model_metrics, conf_mat
