from app import DrawingPredictionApp


if __name__ == "__main__":
    app = DrawingPredictionApp("Digit Recognizer", [str(x) for x in range(10)])
    app.run()