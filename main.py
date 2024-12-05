from app import DrawingPredictionApp


name = "Digit Recognizer"
outputs = [str(x) for x in range(10)] # Set the possible outputs here

if __name__ == "__main__":
    digit_recognizer = DrawingPredictionApp(name, outputs)
    digit_recognizer.run()