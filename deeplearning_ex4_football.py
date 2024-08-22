import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import cv2  
import matplotlib.pyplot as plt
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
    def __init__(self, frame_collections, video_paths, transform=None):
        self.frame_collections = frame_collections
        # self.root_dir = root_dir
        self.video_paths = video_paths
        self.transform = transform
        
        self.frames = []
        self.annotations = []
        self.frame_map = {}  # Map from global idx to (video_index, frame_number)
        
        # Load frames and annotations from each video
        global_idx = 0
        for video_index, frame_collection in enumerate(frame_collections):
            frames = frame_collection['images']
            annotations = frame_collection['annotations']
            for _, frame in enumerate(frames):
                frame_number = frame['id']  # Frame number within the video
                self.frame_map[global_idx] = (video_index, frame_number)
                self.frames.append(frame)
                self.annotations.append(annotations)
                global_idx += 1

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        
        # Get the video index and frame number corresponding to idx
        video_index, frame_number = self.frame_map[idx]
        print (f"HERE IS MY POCCESS {video_index} :::::::, {frame_number}")

        # Load the video file using OpenCV
        video_path = self.video_paths[video_index]
        cap = cv2.VideoCapture(video_path)

        # Set the frame position and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from video {video_path}")

        # Convert the frame (which is in BGR) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = frame_rgb
        # image = Image.fromarray(frame_rgb)

        # Get the frame info at the specified index and details data
        frame_info = self.frames[idx]
        frame_id = frame_info['id']
        annotations_info = self.annotations[video_index]
        jersey_numbers = []

        # Load the image
        # img_name = os.path.join(self.root_dir, frame_info['file_name'])
        # # print (img_name)
        # image = Image.open(img_name).convert('RGB')

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
    transforms.ToTensor(),
    transforms.Resize((1200, 3840)),
])

# Load multiple JSON files
frame_collections = []
json_files = ['Match_1951_1_0_subclip.json', 'Match_1951_1_1_subclip.json']  # Add more JSON files here

for json_file in json_files:
    with open(json_file, 'r') as f:
        frame_collections.append(json.load(f))

# List your video paths here
video_paths = ["Match_1951_1_0_subclip.mp4", "Match_1951_1_1_subclip.mp4"]

# Initialize the dataset
dataset = FootballDataset(frame_collections=frame_collections, video_paths=video_paths, transform=transform)

print(len(dataset))
sample = dataset[0]

# # Print the filtered bounding boxes
# for i, bbox in enumerate(sample['bboxes'], 1):
#   print(f"Bounding box {i}: {bbox}")

# # Print the jersey_number bounding boxes
# for i, jnum in enumerate(sample['jersey_numbers'], 1):
#   print(f"jersey_number player {i}: is {jnum}")

# To display the frame
frame = sample['image']

# Convert the tensor back to a PIL image for display
frame_pil = transforms.ToPILImage()(frame)
frame_pil.show()


# Display using matplotlib
# plt.imshow(frame_pil)
# plt.axis('off')  # Turn off axis
# plt.show()

