# table_module.py

import torch
from torchvision import transforms
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from PIL import Image
import numpy as np
import easyocr
from tqdm.auto import tqdm

class TableExtractor:
    class MaxResize(object):
        """
        Custom transformation to resize the image while maintaining aspect ratio.
        """
        def __init__(self, max_size=800):
            self.max_size = max_size

        def __call__(self, image):
            width, height = image.size
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))
            return resized_image

    def __init__(self, 
                 detection_model_path="microsoft/table-transformer-detection", 
                 structure_model_path="microsoft/table-structure-recognition-v1.1-all",
                 max_resize_size=800,
                 structure_max_resize_size=1000,
                 ocr_lang='en'):
        """
        Initializes the TableExtractor with necessary models and OCR.

        Args:
            detection_model_path (str): Path or identifier for the object detection model.
            structure_model_path (str): Path or identifier for the table structure recognition model.
            max_resize_size (int): Maximum size for resizing during detection.
            structure_max_resize_size (int): Maximum size for resizing during structure recognition.
            ocr_lang (str): Language for OCR.
        """
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load object detection model
        print("Loading object detection model...")
        self.detection_model = AutoModelForObjectDetection.from_pretrained(detection_model_path, revision="no_timm")
        self.detection_model.to(self.device)
        self.detection_model.eval()

        # Load table structure recognition model
        print("Loading table structure recognition model...")
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(structure_model_path)
        self.structure_model.to(self.device)
        self.structure_model.eval()

        # Define detection transforms
        self.detection_transform = transforms.Compose([
            self.MaxResize(max_resize_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Define structure transforms
        self.structure_transform = transforms.Compose([
            self.MaxResize(structure_max_resize_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Initialize EasyOCR reader
        print("Initializing OCR...")
        self.reader = easyocr.Reader([ocr_lang], gpu=torch.cuda.is_available())

    def box_cxcywh_to_xyxy(self, x):
        """
        Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2).

        Args:
            x (torch.Tensor): Bounding boxes in (cx, cy, w, h) format.

        Returns:
            torch.Tensor: Bounding boxes in (x1, y1, x2, y2) format.
        """
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        """
        Rescale bounding boxes to image size.

        Args:
            out_bbox (torch.Tensor): Bounding boxes.
            size (tuple): (width, height) of the image.

        Returns:
            torch.Tensor: Rescaled bounding boxes.
        """
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def outputs_to_objects(self, outputs, img_size, id2label):
        """
        Convert model outputs to a list of detected objects.

        Args:
            outputs (dict): Model outputs.
            img_size (tuple): (width, height) of the image.
            id2label (dict): Mapping from label IDs to label names.

        Returns:
            list: List of detected objects with labels, scores, and bounding boxes.
        """
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label.get(int(label), "no object")
            if class_label != 'no object':
                objects.append({
                    'label': class_label,
                    'score': float(score),
                    'bbox': [float(elem) for elem in bbox]
                })

        return objects

    def get_cell_coordinates_by_row(self, table_data):
        """
        Extract cell coordinates organized by rows and columns.

        Args:
            table_data (list): List of detected table elements.

        Returns:
            list: List of rows with their corresponding cell coordinates.
        """
        # Extract rows and columns
        rows = [entry for entry in table_data if entry['label'] == 'table row']
        columns = [entry for entry in table_data if entry['label'] == 'table column']

        # Sort rows and columns by their Y and X coordinates, respectively
        rows.sort(key=lambda x: x['bbox'][1])
        columns.sort(key=lambda x: x['bbox'][0])

        # Function to find cell coordinates
        def find_cell_coordinates(row, column):
            cell_bbox = [
                column['bbox'][0],
                row['bbox'][1],
                column['bbox'][2],
                row['bbox'][3]
            ]
            return cell_bbox

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

            # Sort cells in the row by X coordinate
            row_cells.sort(key=lambda x: x['column'][0])

            # Append row information to cell_coordinates
            cell_coordinates.append({
                'row': row['bbox'],
                'cells': row_cells,
                'cell_count': len(row_cells)
            })

        # Sort rows from top to bottom
        cell_coordinates.sort(key=lambda x: x['row'][1])

        return cell_coordinates

    def apply_ocr(self, cell_coordinates, image):
        """
        Apply OCR to each cell in the table.

        Args:
            cell_coordinates (list): List of rows with cell coordinates.
            image (PIL.Image): The original image.

        Returns:
            dict: Extracted table data organized by rows and columns.
        """
        data = {}
        max_num_columns = 0
        for idx, row in enumerate(tqdm(cell_coordinates, desc="Applying OCR")):
            row_text = []
            for cell in row["cells"]:
                # Crop cell out of image
                cell_image = image.crop(cell["cell"])
                cell_image_np = np.array(cell_image)
                # Apply OCR
                result = self.reader.readtext(cell_image_np)
                if len(result) > 0:
                    text = " ".join([x[1] for x in result])
                    row_text.append(text)
                else:
                    row_text.append("")  # Empty string for cells with no text

            if len(row_text) > max_num_columns:
                max_num_columns = len(row_text)

            data[idx] = row_text

        print("Max number of columns:", max_num_columns)

        # Pad rows which don't have max_num_columns elements
        for row, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + [""] * (max_num_columns - len(row_data))
                data[row] = row_data

        return data

    def format_table_as_markdown(self, table_data):
        """
        Convert table data to Markdown format.

        Args:
            table_data (dict): Extracted table data organized by rows and columns.

        Returns:
            str: Table in Markdown format.
        """
        if not table_data:
            return "No table data found."

        # Determine number of columns
        num_columns = max(len(row) for row in table_data.values())

        # Create header
        header = "| " + " | ".join([f"Column {i+1}" for i in range(num_columns)]) + " |"
        separator = "| " + " | ".join(["---"] * num_columns) + " |"

        # Initialize markdown table
        markdown_table = [header, separator]

        # Append each row
        for row_idx in sorted(table_data.keys()):
            row = table_data[row_idx]
            row_markdown = "| " + " | ".join(row) + " |"
            markdown_table.append(row_markdown)

        return "\n".join(markdown_table)

    def extract_markdown(self, image):
        """
        Extracts table data from an image and returns it in Markdown format.

        Args:
            image (PIL.Image): The input image containing the table.

        Returns:
            str: Extracted table in Markdown format.
        """
        # Apply detection transforms
        transformed_image = self.detection_transform(image)
        pixel_values = transformed_image.unsqueeze(0).to(self.device)

        # Forward pass for detection model
        with torch.no_grad():
            outputs = self.detection_model(pixel_values)

        # Update id2label to include "no object"
        id2label = self.detection_model.config.id2label.copy()
        id2label[len(id2label)] = "no object"

        # Convert outputs to objects
        objects = self.outputs_to_objects(outputs, image.size, id2label)
        print(f"Detected {len(objects)} objects.")

        # Forward pass for structure model
        transformed_structure_image = self.structure_transform(image)
        structure_pixel_values = transformed_structure_image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            structure_outputs = self.structure_model(structure_pixel_values)

        # Update structure id2label to include "no object"
        structure_id2label = self.structure_model.config.id2label.copy()
        structure_id2label[len(structure_id2label)] = "no object"

        # Convert structure outputs to objects
        structure_objects = self.outputs_to_objects(structure_outputs, image.size, structure_id2label)
        print(f"Detected {len(structure_objects)} structure objects.")

        # Get cell coordinates
        cell_coordinates = self.get_cell_coordinates_by_row(structure_objects)

        # Apply OCR
        print("Applying OCR to detected cells...")
        table_data = self.apply_ocr(cell_coordinates, image)

        # Convert table data to Markdown
        markdown_table = self.format_table_as_markdown(table_data)
        print("\nExtracted Table in Markdown Format:\n")
        print(markdown_table)

        return markdown_table
