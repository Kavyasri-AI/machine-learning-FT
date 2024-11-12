import os
import json
import pandas as pd
import numpy as np
from string import printable
from sklearn import model_selection
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input, LSTM, Embedding
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# def training(data = ""):
#     def read_data():
#         df = pd.read_csv(data)
#         # Step 1: Convert raw URL string in list of lists where characters that are contained in "printable" are stored encoded as integer
#         url_int_tokens = [
#             [printable.index(x) + 1 for x in url if x in printable] for url in df.url]
#         # Step 2: Cut URL string at max_len or pad with zeros if shorter
#         max_len = 75
#         X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
#         # Step 3: Extract labels form df to numpy array
#         target = np.array(df.isMalicious)
#         print('Matrix dimensions of X: ', X.shape,
#             'Vector dimension of target: ', target.shape)
#         X_train, X_test, target_train, target_test = model_selection.train_test_split(
#             X, target, test_size=0.25, random_state=33)
#         return X_train, X_test, target_train, target_test

#     X_train, X_test, target_train, target_test = read_data()

#     epochs_num = 10
#     batch_size = 32

#     class SimpleLSTM(object):
#         def __init__(self) -> None:
#             super(SimpleLSTM, self).__init__()
#             self.max_len = 75
#             self.emb_dim = 32
#             self.max_vocab_len = 100
#             self.lstm_output_size = 32
#             self.W_reg = regularizers.l2(1e-4)

#         def build_model(self):
#             main_input = Input(shape=(self.max_len,),
#                             dtype='int32', name='main_input')
#             emb = Embedding(input_dim=self.max_vocab_len, output_dim=self.emb_dim, input_length=self.max_len,
#                             embeddings_regularizer=self.W_reg)(main_input)
#             emb = Dropout(0.2)(emb)
#             lstm = LSTM(self.lstm_output_size)(emb)
#             lstm = Dropout(0.5)(lstm)
#             output = Dense(1, activation='sigmoid', name='output')(lstm)
#             model = Model(inputs=[main_input], outputs=[output])
#             adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999,
#                         epsilon=1e-08, decay=0.0)
#             model.compile(optimizer=adam, loss='binary_crossentropy',
#                         metrics=['accuracy'])
#             return model

#     model_name = "simple_lstm.h5"
#     model = SimpleLSTM().build_model()
#     model.fit(X_train, target_train,
#                 epochs=epochs_num, batch_size=batch_size)
#     model.save(model_name)
#     # del model
#     # model = load_model('/kaggle/input/url-data/simple_lstm.h5')
#     def evaluate_result(y_true, y_pre):
#         accuracy = accuracy_score(y_true, y_pre)
#         precision = precision_score(y_true, y_pre)
#         recall = recall_score(y_true, y_pre)
#         f1 = f1_score(y_true, y_pre)
#         auc = roc_auc_score(y_true, y_pre)

#         print("Accuracy Score is: ", accuracy)
#         print("Precision Score is :", precision)
#         print("Recall Score is :", recall)
#         print("F1 Score: ", f1)
#         print("AUC Score: ", auc)
        
#     def to_y(labels):
#         y = []
#         for i in range(len(labels)):
#             label = labels[i]
#             if label < 0.5:
#                 y.append(0)
#             else:
#                 y.append(1)
#         return y
    
#     y_pred = model.predict(X_test)
#     pred = to_y(y_pred)
#     # print(pred)
#     print(classification_report(target_test, pred, digits=5))
#     evaluate_result(target_test, pred)

#     return classification_report

# dir_path = os.path.dirname(__file__)
# training(data = os.path.join(dir_path,"data/classification_dataset.txt"))

