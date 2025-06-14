import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/utkarshgoel13/Downloads/customer_churn.csv')

print(data.head())
print(data.info())

data = data.drop(columns=['CustomerID']) #Simplify the dataset
data = data.dropna()  # Simple example of dropping missing values

data = pd.get_dummies(data, drop_first=True) #Encoding categorical variables

X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = nn.functional.dropout(x, 0.5, training=self.training)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = ChurnModel()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified example)
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train.values).float())
    loss = criterion(outputs.squeeze(), torch.tensor(y_train.values).float())
    loss.backward()
    optimizer.step()
    
model.eval()
outputs = model(torch.tensor(X_test.values).float())
predictions = (outputs.squeeze().detach().numpy() > 0.5).astype(int)
accuracy = np.mean(predictions == y_test.values)
print(f'Test accuracy: {accuracy}')

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(model.state_dict(), 'churn_model.pth')