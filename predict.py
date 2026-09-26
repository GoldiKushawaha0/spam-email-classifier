import joblib

# Load trained model
model = joblib.load("spam_model.pkl")

print("=== Spam Message Classifier ===")

while True:
    message = input("\nEnter your message (or type 'exit' to stop): ")

    if message.lower() == "exit":
        print("Program stopped.")
        break

    prediction = model.predict([message])[0]

    if prediction == 1:
        print("Result: 🚨 SPAM")
    else:
        print("Result: ✅ HAM (Not Spam)")