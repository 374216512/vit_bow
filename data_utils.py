from torchvision import transforms, datasets
from torch.utils.data import Dataset
import os
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.img_list = []
        self.w_list = []
        self.label_list = []
        self.label2idx = {}
        self.idx2label = {}
        self.weather2idx = {
            "暴风雪": 1,
            "暴雪": 2,
            "暴雨": 3,
            "冰雹": 4,
            "大风": 5,
            "大雾": 6,
            "大雪": 7,
            "大雨": 8,
            "地震": 9,
            "冻雨": 10,
            "多云": 11,
            "风": 12,
            "狂风": 13,
            "雷电": 14,
            "雷雨": 15,
            "雷阵雨": 16,
            "其他": 17,
            "晴": 18,
            "晴朗": 19,
            "晴天": 20,
            "沙尘": 21,
            "沙尘暴": 22,
            "霜冻": 23,
            "台风": 24,
            "雾": 25,
            "小雪": 26,
            "小雨": 27,
            "雪": 28,
            "阴": 29,
            "阴天": 30,
            "雨": 31,
            "雨夹雪": 32,
            "雨天": 33,
            "雨雪": 34,
            "阵雪": 35,
            "阵雨": 36,
            "中雪": 37,
            "山火": 100
        }

        self.final_weather = {
            "暴风雪": {'暴风雪'},
            "暴雪": {"暴雪"},
            "大雪": {"大雪"},
            "小雪": {"小雪"},
            "雪": {"雪"},
            "阵雪": {"阵雪"},
            "中雪": {"中雪"},
            "暴雨": {"暴雨"},
            "大雨": {"大雨"},
            "冻雨": {"冻雨"},
            "小雨": {"小雨"},
            "雨": {"雨", "雨天"},
            "阵雨": {"阵雨"},
            "冰雹": {"冰雹", "霜冻"},

            "大风": {"大风"},
            "狂风": {"狂风"},
            "沙尘": {"沙尘", "沙尘暴"},
            "台风": {"台风"},
            "风": {"风"}, 

            "雾": {"大雾", "雾"},
            "地震": {"地震", },

            # "晴": {"多云", "晴", "晴朗", "晴天", "阴", "阴天"},
            "晴": {"晴", "晴朗", "晴天"},
            "阴": {"多云", "阴", "阴天"},
            "雷电": {"雷电"},
            "雷雨": {"雷雨", "雷阵雨"},
            "其他": {"其他"},
            "雨夹雪": {"雨夹雪", "雨雪"}
        }
        self.final_weather2idx = {
            k: i for i, (k, v) in enumerate(self.final_weather.items())}

        for idx, label in enumerate(sorted(os.listdir(data_dir))):
            c_dir = os.path.join(data_dir, label)
            self.label2idx[label] = idx
            self.idx2label[idx] = label

            for img_path in sorted(os.listdir(c_dir)):
                weather = img_path.split('_')[3]
                self.img_list.append(os.path.join(c_dir, img_path))
                self.w_list.append(weather)
                self.label_list.append(idx)

    def get_weather_idx(self, weather):
        w = None
        for k, v in self.final_weather.items():
            if weather in v:
                w = k
                break

        return self.final_weather2idx[w]

    def __getitem__(self, item):
        img_path = self.img_list[item]
        label = self.label_list[item]
        weather = self.get_weather_idx(self.w_list[item])
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, weather, label

    def __len__(self):
        return len(self.img_list)
