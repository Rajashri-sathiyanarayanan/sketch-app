import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import copy
import os

st.set_page_config(page_title="Ravi Varma Painting Generator 🎨")
st.title("🎨 Ravi Varma Painting Style Transfer")

# Upload your image
uploaded_file = st.file_uploader("Upload your photo (JPEG preferred):", type=['jpg', 'jpeg', 'png'])

# Load the style image
style_path = "ravivarma_style.jpg"
if not os.path.exists(style_path):
    st.error("❌ Style image 'ravivarma_style.jpg' not found. Please place it in the same folder.")
    st.stop()

style_img = Image.open(style_path).convert('RGB')

# Image transform
loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def image_loader(image):
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

# Load the images as tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_tensor = image_loader(style_img).to(device)

# Define ContentLoss and StyleLoss classes
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Load VGG19 model
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Normalization module
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Get the style model
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

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
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i+1]

    return model, style_losses, content_losses

# Run style transfer
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            run[0] += 1
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

# Convert tensor to PIL
def tensor_to_pil(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

# Perform transfer if image is uploaded
if uploaded_file:
    input_image = Image.open(uploaded_file).convert('RGB')
    st.image(input_image, caption="📸 Original Image", use_column_width=True)

    content_tensor = image_loader(input_image).to(device)
    input_tensor = content_tensor.clone()

    st.info("✨ Applying Ravi Varma style... This might take a minute ⏳")

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_tensor, style_tensor, input_tensor)

    output_image = tensor_to_pil(output)
    st.image(output_image, caption="🖼 Styled as Ravi Varma Painting", use_column_width=True)

    # Download button
    output_image.save("styled_output.jpg")
    with open("styled_output.jpg", "rb") as file:
        st.download_button(label="📥 Download Image",
                           data=file,
                           file_name="ravi_varma_styled.jpg",
                           mime="image/jpeg")
