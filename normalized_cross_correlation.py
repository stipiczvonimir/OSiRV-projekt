import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse

SHAPES = ["circle", "square", "star", "triangle"]
SHAPE_COLORS = {
    "circle": "orange",
    "square": "blue",
    "star": "green",
    "triangle": "red"
}

def plot_bbox(bbox_XYXY, label, color, added_labels, score):
    xmin, ymin, xmax, ymax = bbox_XYXY
    plt.plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color=color, label=str(label) if label not in added_labels else "")
    plt.text(xmin, ymin - 5, f"{label} ({score:.2f})", color=color, fontsize=9, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    added_labels.add(label)

def get_images(folder):
    png_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files    

def select_images(base_folder, min_shapes=4, max_shapes=7):
    selected_images = []
    shape_image_map = {}
    for shape in SHAPES:
        shape_folder = os.path.join(base_folder, shape)
        if os.path.exists(shape_folder):
            png_images = get_images(shape_folder)
            if len(png_images) > 0:
                shape_image_map[shape] = png_images[0]  # Edit for different shape images
            else:
                print(f"Not enough images {shape_folder}")
        else:
            print(f"Missing directory for shape: {shape}")
    if not shape_image_map:
        return []
    
    num_shapes = random.randint(min_shapes, max_shapes)
    for _ in range(num_shapes):
        shape = random.choice(list(shape_image_map.keys()))
        selected_images.append(shape_image_map[shape])
    return selected_images

def remove_white_background(img):
    datas = img.getdata()
    new_data = [(255, 255, 255, 0) if item[:3] == (255, 255, 255) else item for item in datas]
    img.putdata(new_data)
    return img

def generate_image(image_paths, background_size=(800, 600)):
    bg = Image.new('RGBA', background_size, (255, 255, 255, 255))

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGBA')
        img = remove_white_background(img)

        x = random.randint(0, background_size[0] - img.width)
        y = random.randint(0, background_size[1] - img.height)

        bg.paste(img, (x, y), img)

    return bg.convert("RGB")

def load_templates(template_dir):
    templates = {}

    for shape in SHAPES:
        shape_dir = os.path.join(template_dir, shape)
        if os.path.exists(shape_dir):
            shape_images = [f for f in os.listdir(shape_dir) if f.endswith(".png")]
            if shape_images:
                selected_image = shape_images[1] if len(shape_images) > 1 else shape_images[0]
                image_path = os.path.join(shape_dir, selected_image)

                img = Image.open(image_path).convert('RGBA')
                img_np = np.array(img)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
                _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                templates[shape] = (gray, mask)
            else:
                print(f"No images found in {shape_dir}")
        else:
            print(f"Missing directory for shape: {shape}")

    return templates

def detect_shapes(image, templates, threshold=0.4):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_boxes, detected_labels, detected_scores = [], [], []
    for label, (template, mask) in templates.items():
        result = cv2.matchTemplate(image_gray, template, cv2.TM_CCORR_NORMED, mask=mask)
        h, w = template.shape
        locations = np.where(result >= threshold)
        for pt in zip(*locations[::-1]):
            new_box = [pt[0], pt[1], pt[0] + w, pt[1] + h]
            score = result[pt[1], pt[0]]
            detected_boxes.append(new_box)
            detected_labels.append(label)
            detected_scores.append(score)

    return detected_boxes, detected_labels, detected_scores

def non_maximum_suppression(boxes, scores, labels, iou_threshold=0.9):
    if len(boxes) == 0:
        return [], [], []
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    while len(order) > 0:
        i = order[0]
        keep_boxes.append(boxes[i])
        keep_scores.append(scores[i])
        keep_labels.append(labels[i])    

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        remaining_indices = np.where(iou <= iou_threshold)[0] + 1
        order = order[remaining_indices]

    return keep_boxes, keep_scores, keep_labels

def main(num_images, min_shapes, max_shapes):
    input_folder = "./shapes/"
    background_size = (800, 600)
    templates = load_templates(input_folder)

    for i in range(num_images):
        print(f"\n--- Image {i+1} ---")
        selected_images = select_images(input_folder, min_shapes=min_shapes, max_shapes=max_shapes)
        if not selected_images:
            print("No selected images")
            continue

        composed_pil = generate_image(selected_images, background_size)
        composed_rgb = np.array(composed_pil)
        composed_bgr = cv2.cvtColor(composed_rgb, cv2.COLOR_RGB2BGR)

        detected_boxes, detected_labels, detected_scores = detect_shapes(composed_bgr, templates)
        final_boxes, final_scores, final_labels = non_maximum_suppression(detected_boxes, detected_scores, detected_labels)

        plt.figure(figsize=(10, 7))
        plt.imshow(composed_rgb)
        added_labels = set()
        for bbox, shape, score in zip(final_boxes, final_labels, final_scores):
            color = SHAPE_COLORS.get(shape, "white")
            plot_bbox(bbox, shape, color, added_labels, score)

        plt.title(f"Detected Shapes (Image {i+1})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='gray', framealpha=1.0)
        plt.tight_layout()
        # plt.savefig("output.jpg", bbox_inches='tight', dpi=300)

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and detect shapes.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--min_shapes", type=int, default=7, help="Minimum number of shapes per image")
    parser.add_argument("--max_shapes", type=int, default=10, help="Maximum number of shapes per image")
    args = parser.parse_args()

    main(args.num_images, args.min_shapes, args.max_shapes)
