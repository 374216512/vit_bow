import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from pytorch_pretrained_vit import MultiModalModel
from PIL import Image

from torch.nn import functional as F
import numpy as np
import pandas as pd

import os

from sklearn.metrics import accuracy_score, confusion_matrix


def parse_args():
    parser = ArgumentParser(description='Multi-modal model')

    parser.add_argument(
        '--path', type=str, default='/home/tyl/code/bow/ceshi/7.7with_weather')
    parser.add_argument('--model_path', type=str,
                        default='/home/tyl/code/bow/vit/PyTorch-Pretrained-ViT/output/2024.07.30_0710上交_256_B32/checkpoint_33.pth.tar')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='B_32',
                        help='model architecture (default: resnet18)')

    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.path

    label2chinese = {
        '1001': '线路断线（完全断线）',
        '1002': '线路断线（子导线断线）',
        '1003': '风偏',
        '1004': '覆冰（绝缘子伞裙桥接）',
        '1005': '覆冰（脱冰跳跃）',
        '1006': '雷击',
        '1007': '山火',
        '1008': '污闪（鸟粪）',
        '1009': '污闪（其他）',
        '1010': '异物（高阻）',
        '1011': '异物（低阻）',
        '1012': '其他'
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict = torch.load(args.model_path)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 定义模型
    model = MultiModalModel(
        num_embeddings=len(state_dict["weather2idx"]),
        weather_embedding_dim=state_dict["embedding_dim"],
        name=state_dict['arch'],
    )
    model.load_state_dict(state_dict['state_dict'], 'cuda')
    model = model.to('cuda')

    def read_txt_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # 去掉每行的换行符，并将每行作为一个列表元素返回
            lines = [line.strip() for line in lines]
        return lines

    # 示例使用
    excel_true_label = []
    file_path = './truelable.txt'
    lines = read_txt_file(file_path)
    # print(lines)
    excel_true_label = lines
    print(excel_true_label)

    all_pred_labels = []

    # 训练
    score = 0
    model.eval()
    data = []
    weather2idx = state_dict['weather2idx']
    idx2label = state_dict['idx2label']
    for i, case in enumerate(sorted(os.listdir(data_path))):
        case_dir = os.path.join(data_path, case)

        case_info = []
        prob_list = []
        true_labels = []
        for img_name in os.listdir(case_dir):
            weather = img_name.split('_')[3]
            true_label = img_name.split('_')[1]
            try:
                weather_idx = weather2idx[weather]
                image = Image.open(os.path.join(case_dir, img_name))
                image = test_transform(image)[None]

                image = image.to(device)
                weather_idx = torch.tensor(weather_idx)[None].to(device)

                _, _, final_pred = model(image, weather_idx)

                # prob = F.softmax(final_pred, dim=-1)
                prob = F.sigmoid(final_pred)
                prob_list.append(prob.cpu().detach())
                true_labels.append(true_label)
            except:
                image = Image.open(os.path.join(case_dir, img_name))
                image = test_transform(image)[None]

                image = image.to(device)

                final_pred = model.image_model(image)[0]

                # prob = F.softmax(final_pred, dim=-1)
                prob = F.sigmoid(final_pred)
                prob_list.append(prob.cpu().detach())
                true_labels.append(true_label)

        p1, p2 = prob_list
        p, true_label = (p1, true_labels[0]) if torch.max(
            p1) >= torch.max(p2) else (p2, true_labels[1])
        top2_probs, top2_indices = torch.topk(p, 2, dim=1)
        label1 = idx2label[top2_indices[0, 0].item()]
        label2 = idx2label[top2_indices[0, 1].item()]
        case_info.extend(
            [case,
                true_label,
                label2chinese[label1],
                round(top2_probs[0, 0].item(), 4),
                label2chinese[label2],
                round(top2_probs[0, 1].item(), 4)]
        )

        data.append(case_info)

        predicted_labels = [label2chinese[label1], label2chinese[label2]]
        all_pred_labels.append(predicted_labels)

    total_score = calculate_score(excel_true_label, all_pred_labels)
    print(f"Total Score: {total_score}")

    columns = ['案例编号', '真实类别', '故障原因1', '概率1', '故障原因2', '概率2']
    data = pd.DataFrame(data, columns=columns)
    data.to_csv('with_weather.csv', index=False, encoding='gbk')


def calculate_score(excel_true_label, predicted_labels):
    total_score = 0

    def parse_label(label):
        """解析标签，将一级和二级类别分开"""
        if '（' in label:
            main_category = label.split('（')[0].strip()
            sub_category = label.split('（')[1].strip('）')
        else:
            main_category = label
            sub_category = None
        if main_category == "异物":
            sub_category = None
        return main_category, sub_category

    def calculate_label_score(actual, prediction):
        """计算单个预测标签的得分"""
        actual_main, actual_sub = parse_label(actual)
        pred_main, pred_sub = parse_label(prediction)

        if actual_main == pred_main:
            if actual_sub == pred_sub:
                return 1
            elif actual_sub is None and pred_sub is None:
                return 1
            else:
                return 0.5
        else:
            return 0

    for actual, predictions in zip(excel_true_label, predicted_labels):
        best_score = 0

        for prediction in predictions:  # 只考虑前两个预测标签
            score = calculate_label_score(actual, prediction)
            if score > best_score:
                best_score = score

        total_score += best_score

    return total_score

    # 计算总分

    # pred_label1 = label2chinese[label1].split("（")[0]
    # pred_label2 = label2chinese[label2].split("（")[0]
    # if excel_true_label[i] == pred_label1 or excel_true_label[i] == pred_label2:
    # score += 1
    # print(f"总得分为:{score}")
if __name__ == '__main__':
    main()
