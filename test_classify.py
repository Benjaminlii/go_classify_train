from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch
import numpy as np

model = None

def get_model():
    global model
    if model == None:
        model = models.resnet50(pretrained=False)
        fc_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_inputs, 219)
        model = model.cpu()
        # 加载训练好的模型
        checkpoint = torch.load('model_best_checkpoint_resnet50.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    return model


def padding_black(img):
    w, h = img.size
    scale = 224. / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    size_fg = img_fg.size
    size_bg = 224
    img_bg = Image.new("RGB", (size_bg, size_bg))
    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))
    img = img_bg
    return img


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x


transform_valid = transforms.Compose([
    transforms.Resize((224, 224), interpolation=2),
    transforms.ToTensor()
])


def do_classify(image_path):
    # 加载模型
    model = get_model()
    # 读取图片
    image = Image.open(image_path)
    # 处理
    image = image.convert('RGB')
    image = padding_black(image)
    # 预测
    image = transform_valid(image).unsqueeze(0)
    image = image.cpu()
    pred = model(image)
    pred = pred.data.cpu().numpy()[0]
    score = softmax(pred)
    pred_id = np.argmax(score)
    return pred_id

if __name__ == "__main__":
    print(type(do_classify("垃圾图片库/其他垃圾_牙刷/img_牙刷_27.jpeg")))
