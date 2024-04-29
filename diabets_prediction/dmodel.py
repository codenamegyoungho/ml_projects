#%%
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
from torch import nn 
from sklearn.model_selection import train_test_split
df = pd.read_csv("diabetes.csv")

y = df["Outcome"]
X = []
for i,rows in df.iterrows():
    X.append([rows["Pregnancies"],rows["Glucose"],rows["BloodPressure"],rows["SkinThickness"],rows["Insulin"],rows["BMI"],rows["DiabetesPedigreeFunction"],rows["Age"]])
X = torch.tensor(X)
X = X.float()
y = torch.tensor(y)
y = y.float()
trainx,testx,trainy,testy = train_test_split(X,y,test_size=0.2,random_state=42)
trainx.shape,trainy.shape



class dmodel(nn.Module):
    def __init__(self,input_features,output_features,hidden_units=16):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(nn.Linear(in_features=input_features,out_features=hidden_units),
        nn.Linear(in_features=hidden_units,out_features=32),
        nn.Linear(in_features=32,out_features=hidden_units),
        nn.Linear(in_features=hidden_units,out_features=output_features))
    
    def forward(self,x):
        return self.linear_layer_stack(x)
model4 = dmodel(input_features=8,output_features=1)
trainx = trainx.float()
trainy = trainy.float()
testx = testx.float()
testy = testy.float()
# model4(trainx)


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model4.parameters(),lr=0.01)

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc 

#%%
epochs = 10000
torch.manual_seed(42)

for epoch in range(epochs):
    y_logits = model4(trainx)
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits.squeeze(),trainy)
    acc = accuracy_fn(y_true=trainy,
                      y_pred = y_pred.squeeze())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model4.eval()
    with torch.inference_mode():
        test_logits = model4(testx)
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits.squeeze(),testy)
        test_acc = accuracy_fn(y_true=testy,
                               y_pred = test_pred.squeeze())
    
    if epoch % 100 == 0:
        print(f"Epoch : {epoch} | Loss : {loss:.5f}, Accuracy: {acc:.2f}%|Test loss : {test_loss :.5f}, Test acc : {test_acc:.2f}")

# %%
