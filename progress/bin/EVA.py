
import os
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, confusion_matrix
from progress.densenet_baseline import MODELS
import matplotlib.pyplot as plt

import seaborn as sns

def predict_patch(patch_path, model):
    patch = Image.open(patch_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    patch = transform(patch)
    patch = patch.unsqueeze(0)  # Adds a batch dimension: (1, num_channels, height, width)
    patch = patch.unsqueeze(0)  # Adds a num_patches dimension: (1, 1, num_channels, height, width)

    output = model(patch)
    output_probabilities = torch.sigmoid(output)

    return output_probabilities.item()




model = MODELS['densenet121'](num_classes=1)
checkpoint = torch.load('F:/Dissertation/BASELINESAVE_PATH628/best.ckpt', map_location=torch.device('cpu'))  # replace with your actual path
model.load_state_dict(checkpoint['state_dict'])
model.eval()


tumor_dir = 'F:/TEST_TUMOR'
healthy_dir = 'F:/TEST_NORMAL'


tumor_probs = [predict_patch(os.path.join(tumor_dir, fname), model) for fname in os.listdir(tumor_dir) if fname.endswith('.png')]
healthy_probs = [predict_patch(os.path.join(healthy_dir, fname), model) for fname in os.listdir(healthy_dir) if fname.endswith('.png')]


y_true = [1]*len(tumor_probs) + [0]*len(healthy_probs)
y_scores = tumor_probs + healthy_probs


fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)



y_pred = [1 if prob >= 0.2 else 0 for prob in y_scores]
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print('AUC:', roc_auc)
print("Baselinewithout Stain Norm")
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)


#plt.figure(figsize=(10,7))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#plt.xlabel('Predicted')
#plt.ylabel('Truth')
#plt.title('Confusion Matrix')
#plt.show()

