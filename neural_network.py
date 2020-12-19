from prepare_data import prepare_data

data = prepare_data()

X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
X_valid, y_valid = data["X_valid"], data["y_valid"]


print("INPUT SHAPE:",X_train.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model=Sequential()
model.add(Dense(44,activation="relu"))
model.add(Dense(22,activation="relu"))
model.add(Dropout(0.6))
model.add(Dropout(0.6))
model.add(Dense(1))

model.compile(loss="mse",optimizer="adam")
model.fit(x=X_train,y=y_train,epochs=50,batch_size=64,validation_data=(X_valid,y_valid),verbose=1)


from sklearn.metrics import mean_absolute_error

prediction_array=model.predict(X_test)

print("ERROR VALUE:",mean_absolute_error(y_test,prediction_array)*0.01)
print("SCORE ON THE TEST SET:",model.evaluate(X_test,y_test))


