def predict_image(model, image):
    output = model(image)
    output = torch.argmax(output, dim=1).item()
    return output
