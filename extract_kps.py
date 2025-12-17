
import json

def extract_kps():
    with open('anns.json', 'r') as f:
        data = json.load(f)
    
    # Find image id for samples/000011.jpg
    img_id = None
    for img in data['images']:
        if '000011.jpg' in img['file_name']:
            img_id = img['id']
            break
            
    if img_id is None:
        print("Image 000011.jpg not found")
        return

    # Find annotations
    for ann in data['annotations']:
        if ann['image_id'] == img_id:
            print(json.dumps(ann['keypoints']))
            return

if __name__ == '__main__':
    extract_kps()
