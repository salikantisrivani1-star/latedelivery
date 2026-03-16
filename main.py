import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")

# Select useful features
data = df[[
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Shipping Mode',
    'Order Item Quantity',
    'Order Item Product Price',
    'Late_delivery_risk'
]]

# Handle missing values
data = data.dropna()

# Encode shipping mode
encoder = LabelEncoder()
data['Shipping Mode'] = encoder.fit_transform(data['Shipping Mode'])

# Features and target
X = data.drop('Late_delivery_risk', axis=1)
y = data['Late_delivery_risk']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("Model saved successfully!")
