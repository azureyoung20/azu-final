
import torch
from torchvision import transforms
from PIL import Image
from progress.densenet_crf import MODELS  # Assuming this is the correct import for your model
import torchvision.models as models
def predict_patch(patch_path, model_path, cfg_path):
    # Load the model
    model = models.densenet121(num_classes=1)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # replace with your actual path
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load and preprocess the patch
    patch = Image.open(patch_path)  # replace with your actual path
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    patch = transform(patch)
    patch = patch.unsqueeze(0)  # Add an extra dimension for the batch size

    # Pass the patch through the model
    output = model(patch)

    # The output is likely to be a tensor with the raw output of the model.
    # You may want to apply a sigmoid or softmax function to convert these into probabilities.
    output_probabilities = torch.sigmoid(output)

    # If your model is a binary classification model, you can get the predicted class by rounding the output probabilities
    predicted_class = output_probabilities.round()



    return output_probabilities

# Call the function
patch_path = 'F:/testpatch/1.png'
model_path = 'F:/Dissertation/论文撰写/实验结果材料/baseline_crf/best.ckpt'
cfg_path = 'D:/pycharm/NCRF-master/configs/densenet_crf.json'
predicted_class = predict_patch(patch_path, model_path, cfg_path)
print(predicted_class)
