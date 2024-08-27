import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage, transforms
from PIL import Image
from torch.utils.data import DataLoader
import cv2  
import json

class FootballDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        
        self.frames = []
        self.annotations = []
        self.frame_map = {}  # Map from global idx to (video_index, frame_number)
        self.video_paths = []
        
        # Load frames and annotations from each video
        global_idx = 0
        for video_file in os.listdir(self.root_dir):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(self.root_dir, video_file)
                json_path = os.path.join(self.root_dir, video_file.replace('.mp4', '.json'))

                if not os.path.exists(json_path):
                    print(f"Warning: JSON file for {video_file} does not exist. Skipping.")
                    continue

                self.video_paths.append(video_path)

                with open(json_path, 'r') as f:
                    frame_collection = json.load(f)

                frames = frame_collection['images']
                annotations = frame_collection['annotations']

                for frame in frames:
                    frame_number = frame['id']  # Frame number within the video
                    self.frame_map[global_idx] = (len(self.video_paths) - 1, frame_number)
                    self.frames.append(frame)
                    self.annotations.append(annotations)
                    global_idx += 1

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        
        # Get the video index and frame number corresponding to idx
        video_index, frame_number = self.frame_map[idx]

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
        # image = Image.fromarray(frame_rgb)
        image = frame_rgb

        # Get the frame info at the specified index and details data
        frame_info = self.frames[idx]
        frame_id = frame_info['id']
        annotations_info = self.annotations[video_index]

        # Load the image
        # img_name = os.path.join(self.root_dir, frame_info['file_name'])
        # # print (img_name)
        # image = Image.open(img_name).convert('RGB')

        # Extract bounding boxes for the specified category_id and image_id
        bbox = [
            annotation['bbox']
            for annotation in annotations_info
            if annotation['category_id'] == 4 and annotation['image_id'] == frame_id]
        
        if len(bbox) > 0:
            bbox = torch.tensor(bbox, dtype=torch.float32)
        else:
            bbox = torch.zeros((0, 4), dtype=torch.float32)  # In case no bbox is found


        # Extract number of player
        # for annotation in annotations_info:
        #   if "attributes" in annotation and "jersey_number" in annotation["attributes"] and annotation['category_id'] == 4 and annotation['image_id'] == frame_id :
        #     jersey_number = annotation["attributes"]["jersey_number"]
        #     jersey_numbers.append(jersey_number)
        
        jersey_number = [annotation["attributes"]["jersey_number"]
            for annotation in annotations_info
            if "attributes" in annotation and "jersey_number" in annotation["attributes"] and annotation['category_id'] == 4 and annotation['image_id'] == frame_id]

        # Print label 

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'bbox': bbox,
            'jersey_number': jersey_number
            # 'labels': torch.tensor(category_id)
        }

        return image, bbox, jersey_number

if __name__ == "__main__":

    # Define any data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1200, 3840)),
    ])

    root_dir = './data'
    train_dataset = FootballDataset(root_dir=root_dir, split='train',transform=transform)
    test_dataset = FootballDataset(root_dir=root_dir, split='test',transform=transform)

    print(len(train_dataset))
    print(len(test_dataset))

    # sample = dataset[0]
    # frame = sample['image']

    image, bbox, jersey_number = train_dataset[0]
    frame = image
    # print (image.shape )
    # print (bbox)
    # print (jersey_number)

    # # Convert the tensor back to a PIL image for display
    # frame_pil = transforms.ToPILImage()(frame)
    # frame_pil.show()

    # Create DataLoader for train dataset
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,  # Adjust batch size as needed
        shuffle=False,  # Shuffle the data for training
        num_workers=4  # Number of subprocesses to use for data loading
    )

    # Create DataLoader for test dataset
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,  # Adjust batch size as needed
        shuffle=False,  # No need to shuffle test data
        num_workers=4  # Number of subprocesses to use for data loading
    )

    # Example: Iterating through the train DataLoader
    for i, (images, bboxes_batch, jersey_numbers_batch) in enumerate(train_loader):
        print(f"Batch {i+1}")
        print(f"Images shape: {images.shape}")
        print(f"Bounding boxes: {bboxes_batch}")
        print(f"Jersey numbers: {jersey_numbers_batch}")

        # Process your batch here...

        # Break after one batch for demonstration purposes
        if i == 0:
            break

