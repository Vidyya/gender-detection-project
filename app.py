from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved vectorizer and model from disk.
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_name = ""

    if request.method == "POST":
        # Get the name entered by the user from the form.
        input_name = request.form.get("name", "").strip()

        if input_name:
            # Convert the input name using the saved vectorizer.
            name_vector = vectorizer.transform([input_name])
            # Predict gender using the trained model.
            prediction = model.predict(name_vector)[0].title()
        else:
            prediction = "Please enter a name."

    return render_template(
        "index.html",
        prediction=prediction,
        input_name=input_name,
    )

if __name__ == "__main__":
    # Run the Flask development server on localhost port 5000.
    app.run(debug=True)
