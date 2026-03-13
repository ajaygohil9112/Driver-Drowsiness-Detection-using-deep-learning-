#1. import libraries----------------------------------------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf  
from tensorflow import keras
from keras.applications import InceptionV3 #import Inceptionv3 (transfer learning model) algorithm that already knows how to recognize millions of objects                              
from keras.models import Model #tool used to stitch our custom layers
from keras.layers import Dropout,Input,Flatten,Dense,MaxPooling2D  #neural network building blocks
from keras.optimizers import Adam        #adam algorithm that helps neural network learn from its mistakes and update its weights
from keras.preprocessing.image import ImageDataGenerator   #tool to load pictures and create more study material by slighty altering them
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau  #safety net during training like saving best model adn stopping if it stops learning

#Data Augmentation and Loading
batchsize=2  #tells to look 2 image at a time
train_datagen=ImageDataGenerator(rescale=1./255, rotation_range=0.2,shear_range=0.2, #data preparation happens
                              zoom_range=0.2,width_shift_range=0.2, #standardize pixels, randomly rotate,tilt, shift
                              height_shift_range=0.2, validation_split=0.2)

train_data=train_datagen.flow_from_directory(r'D:\project\project2\Prepared_Data\Train', #point generator to our train folder and resize every image 
                                             target_size=(80,80),batch_size=batchsize,class_mode='categorical',subset='training',color_mode='rgb')
validation_data=train_datagen.flow_from_directory(r'D:\project\project2\Prepared_Data\Train', #20% validation split, assigning it to validation subset 
                                             target_size=(80,80),batch_size=batchsize,class_mode='categorical',subset='validation',color_mode='rgb')
test_datagen=ImageDataGenerator(rescale=1./255)  #rescale the pixels here.
test_data=test_datagen.flow_from_directory(r'D:\project\project2\Prepared_Data\Test',
                                             target_size=(80,80),batch_size=batchsize,class_mode='categorical',color_mode='rgb')

#3. Transfer Learning
base_model=InceptionV3(include_top=False,weights='imagenet',input_tensor=Input(shape=(80,80,3),batch_size=batchsize)) #load pre-trained model
head_model=base_model.output  #grab the output from the inceptionv3 base
head_model=Flatten()(head_model)  #take the 2d picture data and flatten it out into a single 1d line of data 
head_model=Dense(64,activation='relu')(head_model)  #add a fully connected neural network layer with 64 neurons to process flattened data
head_model=Dropout(0.5)(head_model)  #randomly turn off 50% of the neurons during each training step
head_model=Dense(2,activation='softmax')(head_model) #final decision layer it has 2 neurons (one for open and one for closed)

model=Model(inputs=base_model.input,outputs=head_model) #we officially stitch inceptionv3 base annd our custom layer together into one model

for layer in base_model.layers: #lock the weights of the inceptionV3 base
    layer.trainable=False


#Setting the rules & Starting Training
checkpoint=ModelCheckpoint(r'D:\project\project2\models', #save state  (if it gets high score it saves the model to models folder)
                           monitor='val_loss', save_best_only=True,verbose=3)

earlystop=EarlyStopping(monitor='val_loss',patience=7,verbose=3,restore_best_weights=True) #if model takes 7 practice quizzes in row without improving this stops the training
learning_rate=ReduceLROnPlateau(monitor='val_loss',patience=3,verbose=3) #model stucks and stops improving for 3 epochs lowers the learning rate

callbacks=[checkpoint,earlystop,learning_rate]  #bundles safety net rules together

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])  #prepare for model traning

model.fit(train_data,steps_per_epoch=train_data.samples//batchsize,  #tell model start studing the train_data
                    validation_data=validation_data,
                    validation_steps=validation_data.samples//batchsize,
                    callbacks=callbacks,
                    epochs=5)