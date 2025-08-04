import numpy as np

np.set_printoptions(precision=4,floatmode='fixed',suppress=True)

def relu(z):
    return np.maximum(0,z)
def sigmoid(z):
    z = np.clip(z,-500,500)
    return 1/(1+np.exp(-z))

def get_input():
    print("backward propagation - binary classsification")

    x = list(map(float,input().split()))
    x = np.array(x).reshape(1,-1)

    y_true = float(input("Enter the ture label (0 or 1): "))
    y_true = np.array(y_true).reshape(1,-1)

    learning_rate = float(input("Enter the learning rate : "))

    print("Enter the weight : ")
    w1 = list(map(float,input().split()))
    w1 = np.array(w1).reshape(x.shape[1],1)

    print("Enter the bias1")
    b1 = list(map(float,input().split()))
    b1 = np.array(b1).reshape(1,1)

    print("Enter the weight_2")
    w2 = list(map(float,input().split()))
    w2 = np.array(w2).reshape(x.shape[1],1)

    print("Enter the bias_2")
    b2 = list(map(float,input().split()))
    b2 = np.array(b2).reshape(1,1)

    epo = int(input("Enter how many epochs you want : "))

    for epoch in range(epo):
        print(f"epoch {epoch+1}")
        w1,b1,w2,b2 = forward_propagation(x,w1,b1,w2,b2,y_true,learning_rate)

def forward_propagation(x,w1,b1,w2,b2,y_true,learning_rate):
    print("Forward propagation...!!")
    print("Hidden layer...!!!")
    z1 = np.dot(x,w1) + b1
    a1 = sigmoid(z1)

    print("Hidden layer results ")
    print(f"The z1 = {z1} and a1 = {a1}")
    

    print("Hidden layer...!!!")

    z2 = np.dot(a1,w2) + b2
    a2 = sigmoid(z2)

    print(f"\nOutput (Sigmoid):")
    print(f"  z2 = {z2}")
    print(f"  a2 = sigmoid(z2) = {a2}")


    print(f"\nPrediction: {a2}")
    print(f"True label: {y_true}")
    print(f"Error: {(-(y_true * np.log(a2))+((1-y_true)*np.log(1-a2)))}")

    return back_propagation(x,w1,b1,w2,b2,y_true,learning_rate,a2, a1)

def back_propagation(x,w1,b1,w2,b2,y_true,learning_rate,a2, a1):
    print(f"\n=== BACKWARD PROPAGATION ===")

    print(f"\n Hidden layer..")

    dl_dy = -((y_true/a2)+((1-y_true)/(1-a2)))
    dy_dz2 = a2*(1-a2)
    dz2_dw2 = a1
    dz2_db2 = 1
    dl_dw2 = dl_dy * dy_dz2 * dz2_dw2
    dl_db2 = dl_dy * dy_dz2 * dz2_db2

    w2new = w2 - learning_rate * dl_dw2
    b2new = b2 - learning_rate * dl_db2

    print(f"{w2}--->{w2new}")
    print(f"{b1}--->{b2new}")

    print("output layer")

    dz2_da1 = w2
    da1_dz1 = a1 * (1-a1)
    dz1_dw1 = x
    dz1_db1 = 1

    dl_da1 = dl_dy * dy_dz2 * dz2_da1
    dl_dw1 = dl_da1 * da1_dz1 * dz1_dw1
    dl_db1 = dl_da1 * da1_dz1 * dz1_db1

    w1new = w1 - learning_rate * dl_dw1
    b1new = b1 - learning_rate * dl_db1

    print(f"w1 = {w1} --> {w1new}")
    print(f"b1 = {b1} --> {b1new}")
    
    return w1new, b1new, w2new, b2new
if __name__=="__main__":
    get_input()