import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path, size=512):
    image = Image.open(img_path).convert('RGB')
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

content_img = load_image("content.jpg")
style_img = load_image("style.jpg")
input_img = content_img.clone().requires_grad_(True)

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

def get_model_and_losses(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    content_losses = []
    style_losses = []
    model = nn.Sequential().to(device)
    i = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name == "conv_4":
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss", content_loss)
            content_losses.append(content_loss)

        if name in ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses

cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)

optimizer = optim.Adam([input_img], lr=0.01)
num_steps = 300
style_weight = 1e6
content_weight = 1

print("Optimizing...")
for step in range(1, num_steps+1):
    optimizer.zero_grad()
    model(input_img)
    
    style_score = 0
    content_score = 0

    for sl in style_losses:
        style_score += sl.loss
    for cl in content_losses:
        content_score += cl.loss

    loss = style_weight * style_score + content_weight * content_score
    loss.backward()
    optimizer.step()
    input_img.data.clamp_(0, 1)

    if step % 50 == 0 or step == num_steps:
        print(f"Step {step}: Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")

imshow(input_img, title="Stylized Image")
output = transforms.ToPILImage()(input_img.cpu().squeeze(0))
output.save("stylized_output.jpg")
print("Saved as stylized_output.jpg")