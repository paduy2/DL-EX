import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import json

"""
    Giới thiệu bộ dataset football
    - Mỗi 1 video kéo dài khoảng 1 phút, với 25 khung hình 1 giây (25 FPS)
    - Mỗi 1 video sẽ có 1 file annotation .json tương ứng (Mình khuyến khích các bạn mở các file này ra và
    thử cố gắng hiểu về các attribute trong các file này nhé)
    - Đối với từ khóa "categories": Nhìn chung sẽ có 4 đối tượng được annotate trong các video, với ID là 1 cho đến 4. Tạm thời các bạn
    chỉ cần quan tâm đến id = 4, là các cầu thủ là được
    - Đối với từ khóa "images": Đây là thông tin về các frame trong video. Các bạn chú ý là ở đây frame xuất phát từ 1,
    nhưng trong lập trình chỉ số xuất phát từ 0 nhé. Nhìn chung sẽ có 1500 frames, tương ứng với 1 phút
    - Đối với từ khóa "annotations": ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT. Các bạn sẽ thấy trong trường này có rất nhiều
    dictionary, mỗi 1 dictionary tương ứng với 1 object trong 1 frame nhất định, trong đó:
        + id: không cần quan tâm
        + image_id: id của frame (chạy từ 1 cho đến 1500)
        + category_id: Các bạn chỉ cần quan tâm đến những item mà category_id = 4 (player) là được
        
    TASK: Các bạn hãy xây dựng Dataset cho bộ dataset này, với các quy tắc sau:
    - Hàm __init__ tùy ý các bạn thiết kế
    - Hàm __len__ trả về tổng số lượng frame có trong tất cả các video
    - Hàm __getitem__(self, idx) trả về list của các bức ảnh đã được crop về các cầu thủ (trong hầu hết các frame
    là sẽ có 1 cầu thủ trong 1 frame) và list các số áo tương ứng của các cầu thủ này. idx sẽ theo quy tắc sau: Giả sử
    các bạn gộp tất cả các video thành 1 video dài (thứ tự các video con tùy các bạn), thì idx sẽ là index của video 
    dài đó. Ví dụ trong trường hợp chúng ta có 3 video con dài 1 phút, thì video tổng sẽ dài khoảng 3 phút và có
    4500 frames tổng cộng.
    
    GOOD LUCK!
"""

class FootballDataset(Dataset):
    def __init__(self, frame_collection, root_dir, transform=None):
        self.frame_collection = frame_collection
        self.root_dir = root_dir
        self.transform = transform
        self.frames = frame_collection['images']  # Assuming 'frames' is a key in the JSON that holds frame data
        self.annotations = frame_collection['annotations'] # Elements of annotations are the detail of each frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Get the frame info at the specified index and details data
        frame_info = self.frames[idx]
        frame_id = frame_info['id']
        print ("test----------------------------------------------")
        annotations_info = self.annotations
        jersey_numbers = []

        # Load the image
        img_name = os.path.join(self.root_dir, frame_info['file_name'])
        print (img_name)
        image = Image.open(img_name).convert('RGB')

        # Extract bounding boxes for the specified category_id and image_id
        bboxes = [
            annotation['bbox']
            for annotation in annotations_info
            if annotation['category_id'] == 4 and annotation['image_id'] == frame_id]
        
        # Extract number of player
        # for annotation in annotations_info:
        #   if "attributes" in annotation and "jersey_number" in annotation["attributes"] and annotation['category_id'] == 4 and annotation['image_id'] == frame_id :
        #     jersey_number = annotation["attributes"]["jersey_number"]
        #     jersey_numbers.append(jersey_number)
        
        jersey_numbers = [annotation["attributes"]["jersey_number"]
            for annotation in annotations_info
            if "attributes" in annotation and "jersey_number" in annotation["attributes"] and annotation['category_id'] == 4 and annotation['image_id'] == frame_id]

        # Print label 

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'bboxes': bboxes,
            'jersey_numbers': jersey_numbers
            # 'labels': torch.tensor(category_id)
        }

        return sample
    
# Define any data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

with open('Match_1951_1_0_subclip.json', 'r') as f:
    frame_collection = json.load(f)

# Initialize the dataset
dataset = FootballDataset(frame_collection=frame_collection, root_dir='./extracted_frames', transform=transform)

print(len(dataset))
sample = dataset[0]

# Print the filtered bounding boxes
for i, bbox in enumerate(sample['bboxes'], 1):
  print(f"Bounding box {i}: {bbox}")

# Print the jersey_number bounding boxes
for i, jnum in enumerate(sample['jersey_numbers'], 1):
  print(f"jersey_number player {i}: is {jnum}")

# Example usage:
# for data in dataset:
    # print(data['image'].shape, data['boxes'], data['labels'])

