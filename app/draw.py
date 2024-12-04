import tkinter as tk, tkinter.simpledialog as simpledialog
from tkinter import messagebox
from .neural_network import NeuralNetwork
import torch


class DrawingPredictionApp:
    canvas_size = 500
    square_division = 50
    square_size = canvas_size // square_division

    def __init__(self, title, options: list, model_filepath: str = "model.pth", hidden_layers: list[int] = [128, 64]):
        input_size = self.square_size ** 2
        self.nn = NeuralNetwork(input_size, *hidden_layers, len(options))
        self.options = options
        self.model_filepath = model_filepath

        self.ui_setup(title)

        print(f"Input size: {input_size}")
        print(f"Network: {self.nn}")

    def ui_setup(self, title):
        self.window = tk.Tk()
        self.window.title(title)

        self.canvas = tk.Canvas(self.window, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.clear_button = tk.Button(self.window, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.predict_button = tk.Button(self.window, text="Predict", command=self.show_prediction)
        self.predict_button.pack()

        self.predict_frame = tk.Frame(self.window)
        self.predict_frame.pack()

        self.feedback_label = tk.Label(self.predict_frame, text="", font=("Arial", 14))
        self.feedback_label.pack()

        self.actual_answer = tk.Entry(self.predict_frame)
        self.actual_answer.pack()

        self.submit_answer = tk.Button(
            self.predict_frame, 
            text="Submit", 
            command=lambda: self.refine_prediction(self.actual_answer.get())
        )
        self.submit_answer.pack()

        self.model_frame = tk.Frame(self.window)
        self.model_frame.pack()

        self.save_model_button = tk.Button(self.predict_frame, text="Save", command=self.save_model)
        self.save_model_button.pack()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="black", outline="black")

    def clear_canvas(self):
        self.canvas.delete("all")

    def tokenize_canvas(self):
        from PIL import ImageGrab
        x = self.window.winfo_rootx() + self.canvas.winfo_x()
        y = self.window.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        image = ImageGrab.grab((x, y, x1, y1)).convert("L")
        image = image.resize((self.square_size, self.square_size))
        data = list(image.getdata())
        token = [(255 - pixel) / 255 for pixel in data]
        return token
    
    def predict(self):
        token = torch.tensor(self.tokenize_canvas(), dtype=torch.float32).unsqueeze(0)
        print(token, len(token))
        with torch.no_grad():
            output = self.nn(token)
            probabilities = torch.softmax(output, dim=1)
            predicted_index = probabilities.argmax().item()
        return self.options[predicted_index], probabilities[0].tolist()
    
    def show_prediction(self):
        prediction, _ = self.predict()
        self.feedback_label.config(text=f"Prediction: {prediction}")
    
    def refine_prediction(self, ans):
        if ans not in self.options:
            raise ValueError(f"{ans} not a possible option.")
        
        token = torch.tensor(self.tokenize_canvas(), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor([self.options.index(ans)], dtype=torch.long)
        self.nn.train_model(token, label, epochs=10, learning_rate=0.01)
        self.feedback_label.config(text="Model refined with your feedback.")
    
    def save_model(self):
        torch.save(self.nn.state_dict(), self.model_filepath)
        self.feedback_label.config(text=f"Model saved to {self.model_filepath}")

    def load_model(self):
        self.nn.load_state_dict(torch.load(self.model_filepath))
        self.nn.eval()
        self.feedback_label.config(text=f"Model loaded: {self.model_filepath}")

    def run(self):
        self.window.mainloop()