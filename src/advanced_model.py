from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Reshape, Concatenate
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D


from specs import WIDTH, HEIGHT, CHANNEL


class AdvancedModel(Model):
    def __init__(self):
        super(AdvancedModel, self).__init__()
        self.encoder = Sequential([
            Conv2D(8, (5, 5), padding='SAME', 
                input_shape=(HEIGHT, WIDTH, CHANNEL), activation='relu'),
            MaxPooling2D(pool_size=(2, 2), padding='valid'),
    
            Conv2D(16, (5, 5), activation='relu',padding='SAME'),
            MaxPooling2D(pool_size=(2,2), padding='valid'),

            Conv2D(32, (5,5), activation='relu',padding='SAME'),
            MaxPooling2D(pool_size=(2,2), padding='valid'),
        
            Conv2D(64, (5,5), activation='relu',padding='SAME'),
            MaxPooling2D(pool_size=(2,2), padding='valid'),
    
            Flatten()
        ])
        self.decoders = Dense(1024)
        self.decoders = Dropout(0.75)
        
        self.decoders = [Dense(10, activation='softmax') for _ in range(4)]
        self.concat = Concatenate()
        self.reshape = Reshape((4, 10))


    def call(self, x):
        x = self.encoder(x)
        x = self.concat([decoder(x) for decoder in self.decoders])
        x = self.reshape(x)
        return x