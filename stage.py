import streamlit as st
import os
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import hashlib  # For content hashing
import time
import logging
import pytesseract
import gdown
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Streamlit app title
st.title("Document Analyzer with Element Detection and OCR")

# Constants and configurations
ENTITIES_COLORS = {
    "Caption": (200, 150, 100),
    "Footnote": (100, 100, 150),
    "Formula": (150, 120, 100),
    "List-item": (180, 200, 150),
    "Page-footer": (100, 120, 150),
    "Page-header": (120, 150, 140),
    "Picture": (220, 150, 160),
    "Section-header": (100, 180, 170),
    "Table": (160, 170, 170),
    "Text": (100, 170, 220),
    "Title": (200, 130, 100),
    "Unknown": (128, 128, 128),
}
BOX_PADDING = 2

# Define YOLO model path and download link
MODEL_PATH = "models/yolov10x_best.pt"
FILE_ID = "1jTF4xd0Pu7FDFpLTfSGjgTTolZju4_j7"  # Replace with your actual file ID
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading YOLO model from Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            # Download the file using gdown
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("YOLO model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading the model: {e}")
            st.stop()
    return YOLO(MODEL_PATH)

DETECTION_MODEL = load_model()

def draw_box_and_label(image, start_box, end_box, cls, detection_class_conf):
    """
    Draw bounding box and label on the image.
    """
    box_scale_factor = 0.001
    label_scale_factor = 0.5

    line_thickness = max(
        round(box_scale_factor * (image.shape[0] + image.shape[1]) / 2), 1
    )

    cv2.rectangle(
        img=image,
        pt1=start_box,
        pt2=end_box,
        color=ENTITIES_COLORS.get(cls, (128, 128, 128)),
        thickness=line_thickness,
    )

    text = f"{cls} {detection_class_conf:.2f}"
    font_scale = label_scale_factor
    font_thickness = max(line_thickness - 1, 1)

    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=font_thickness
    )

    cv2.rectangle(
        image,
        (start_box[0], start_box[1] - text_h - BOX_PADDING * 2),
        (start_box[0] + text_w + BOX_PADDING * 2, start_box[1]),
        ENTITIES_COLORS.get(cls, (128, 128, 128)),
        thickness=-1,
    )

    cv2.putText(
        image,
        text,
        (start_box[0] + BOX_PADDING, start_box[1] - BOX_PADDING),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=font_thickness,
    )

def merge_nearby_detections(detections, iou_threshold=0.5):
    """
    Merge nearby detections to reduce duplicates.
    """
    merged = []
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    while detections:
        best = detections.pop(0)
        merged.append(best)

        i = 0
        while i < len(detections):
            if calculate_iou(best['coordinates'], detections[i]['coordinates']) > iou_threshold:
                detections.pop(i)
            else:
                i += 1

    return merged

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1 = max(box1['start'][0], box2['start'][0])
    y1 = max(box1['start'][1], box2['start'][1])
    x2 = min(box1['end'][0], box2['end'][0])
    y2 = min(box1['end'][1], box2['end'][1])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1['end'][0] - box1['start'][0]) * (box1['end'][1] - box1['start'][1])
    area2 = (box2['end'][0] - box2['start'][0]) * (box2['end'][1] - box2['start'][1])

    if area1 + area2 - intersection == 0:
        return 0
    iou = intersection / float(area1 + area2 - intersection)
    return iou

def detect_batch(images_with_info):
    """
    Detect elements in all images at once and return their details.
    """
    preprocessed_images = []
    for image, _, _ in images_with_info:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        preprocessed_images.append(image_cv)

    # Run detection on all images at once
    results_list = DETECTION_MODEL.predict(source=preprocessed_images, conf=0.2, iou=0.8)

    # Prepare the results per image
    all_detected_elements = []
    result_images = []

    for idx, (image_info, results) in enumerate(zip(images_with_info, results_list)):
        image_cv = cv2.cvtColor(np.array(image_info[0]), cv2.COLOR_RGB2BGR)
        page_numbers = image_info[1]
        page_boundary = image_info[2]

        boxes = results.boxes
        detected_elements = []

        if len(boxes) == 0:
            result_images.append(Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)))
            all_detected_elements.append(detected_elements)
            continue

        for box in boxes:
            detection_class_conf = box.conf.item()
            cls_index = int(box.cls)
            cls = list(ENTITIES_COLORS.keys())[cls_index] if cls_index < len(ENTITIES_COLORS) else "Unknown"

            start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
            end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

            detected_elements.append({
                "class": cls,
                "confidence": detection_class_conf,
                "coordinates": {
                    "start": start_box,
                    "end": end_box
                },
                "page_number": page_numbers[0] if page_numbers else None  # Simplified for single-page processing
            })

            draw_box_and_label(image_cv, start_box, end_box, cls, detection_class_conf)

        # Merge nearby detections
        detected_elements = merge_nearby_detections(detected_elements)

        # Sort detected elements by their position on the page (top to bottom, left to right)
        detected_elements.sort(key=lambda x: (x['coordinates']['start'][1], x['coordinates']['start'][0]))

        # Assign index based on sorted order
        for element_idx, element in enumerate(detected_elements):
            element['index'] = element_idx

        result_images.append(Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)))
        all_detected_elements.append(detected_elements)

    return result_images, all_detected_elements

def process_pdf(temp_file_path):
    """
    Process a PDF document and return a list of images from each page.
    """
    images = []
    try:
        pdf_document = fitz.open(temp_file_path)
        num_pages = len(pdf_document)
        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Optionally resize the image if it's too large
            max_size = (1200, 1200)
            img.thumbnail(max_size, Image.LANCZOS)
            images.append((img, [page_num], None))  # None for page_boundary in single-page processing
        return images
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return None

def process_image(temp_file_path):
    """
    Process an image file and return a PIL Image.
    """
    try:
        img = Image.open(temp_file_path).convert("RGB")
        # Optionally resize the image if it's too large
        max_size = (1200, 1200)
        img.thumbnail(max_size, Image.LANCZOS)
        return [(img, [0], None)]  # None for page_boundary in single-page processing
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        st.error(f"Error processing image: {str(e)}")
        return None

def crop_and_encode_image(image, element_region):
    """
    Crop the image to the element region and return as PIL Image.
    """
    padding = 5  # 5 pixels padding
    x_start = max(element_region["coordinates"]["start"][0] - padding, 0)
    y_start = max(element_region["coordinates"]["start"][1] - padding, 0)
    x_end = min(element_region["coordinates"]["end"][0] + padding, image.width)
    y_end = min(element_region["coordinates"]["end"][1] + padding, image.height)

    element_image = image.crop((x_start, y_start, x_end, y_end))

    # Maintain aspect ratio while resizing
    max_size = (800, 800)  # Adjusted size
    element_image.thumbnail(max_size, Image.LANCZOS)

    return element_image  # Return PIL Image object

def process_chunk(image, chunk):
    """
    Process a single chunk of detected elements.
    """
    results = []

    for element in chunk['elements']:
        if element['class'] == 'Table':
            # Process as a table with TableExtractor (if available)
            result = process_element_with_table_extraction(image, element)
        elif element['class'] == 'Picture':
            # For pictures, just extract and display the image
            result = process_element_with_image(image, element)
        else:
            # Process with pytesseract OCR
            result = process_element_with_ocr(image, element)
        results.append(result)

    # Aggregate results for the chunk
    return {
        "chunk_index": chunk['chunk_index'],
        "elements": results
    }

def process_element_with_image(image, element):
    """
    Process an element by extracting and returning the image.
    """
    try:
        cropped_image = crop_and_encode_image(image, element)
    except Exception as e:
        logging.error(f"Error processing image element: {str(e)}")
        cropped_image = None

    result = {
        "element_index": element['index'],
        "class": element['class'],
        "result": cropped_image,  # This will be an Image object
        "coordinates": element["coordinates"]
    }

    return result

def process_element_with_ocr(image, element):
    """
    Process an element using pytesseract OCR.
    """
    try:
        cropped_image = crop_and_encode_image(image, element)
        text = pytesseract.image_to_string(cropped_image)
    except Exception as e:
        text = f"Error processing element with OCR: {str(e)}"
        logging.error(f"Error processing element with OCR: {str(e)}")

    result = {
        "element_index": element['index'],
        "class": element['class'],
        "result": text.strip(),
        "coordinates": element["coordinates"]
    }

    return result

def process_element_with_table_extraction(image, element):
    """
    Process an element as a table using TableExtractor.
    """
    try:
        cropped_image = crop_and_encode_image(image, element)
        # Replace the following line with actual TableExtractor code if available
        markdown_table = "Extracted table data (placeholder)"
        # If you have TableExtractor, use:
        # markdown_table = table_extractor.extract_markdown(cropped_image)
    except Exception as e:
        markdown_table = f"Error processing table: {str(e)}"
        logging.error(f"Error processing table: {str(e)}")

    result = {
        "element_index": element['index'],
        "class": element['class'],
        "result": markdown_table,
        "coordinates": element["coordinates"]
    }

    return result

def intelligent_chunking(detected_elements, max_chunk_size=5):
    """
    Group detected elements into chunks.
    """
    chunks = []
    current_chunk_elements = []

    for element in detected_elements:
        current_chunk_elements.append(element)
        if len(current_chunk_elements) >= max_chunk_size:
            chunks.append({'elements': current_chunk_elements})
            current_chunk_elements = []

    if current_chunk_elements:
        chunks.append({'elements': current_chunk_elements})

    # Assign chunk indices
    for idx, chunk in enumerate(chunks):
        chunk['chunk_index'] = idx

    return chunks

def main():
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"])

    if uploaded_file is not None:
        # Save uploaded file and generate file_id
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Generate a file ID
        file_id = hashlib.md5((uploaded_file.name + str(os.path.getmtime(temp_file_path))).encode()).hexdigest()

        file_ext = os.path.splitext(temp_file_path)[1].lower()
        images_with_info = []

        if file_ext == ".pdf":
            images_with_info = process_pdf(temp_file_path)
            if images_with_info is None:
                return
            num_pages = len(images_with_info)
            st.write(f"The PDF has {num_pages} pages.")
        elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            images_with_info = process_image(temp_file_path)
            if images_with_info is None:
                return
        else:
            st.error("Unsupported file type")
            return

        all_results = []

        # Detect elements in all images at once
        st.info("Running batch detection on all images...")
        try:
            result_images, detected_elements_list = detect_batch(images_with_info)
        except Exception as e:
            logging.error(f"Error during batch detection: {str(e)}")
            st.error(f"Error during batch detection: {str(e)}")
            return

        # Process each image and its detected elements individually
        for idx, (image_info, detected_elements) in enumerate(zip(images_with_info, detected_elements_list)):
            image = image_info[0]
            st.write(f"Processing image {idx+1}/{len(images_with_info)} with {len(detected_elements)} detected elements.")

            # Display the image with detections
            st.image(result_images[idx], caption=f"Image {idx+1} with Detections", use_column_width=True)

            if detected_elements:
                # Apply the chunking algorithm
                chunks = intelligent_chunking(detected_elements, max_chunk_size=5)

                # Process chunks sequentially
                chunk_results = []
                for chunk in chunks:
                    result = process_chunk(image, chunk)
                    chunk_results.append(result)
                all_results.extend(chunk_results)
            else:
                st.write(f"No detected elements in image {idx+1}.")
                continue

        # Display results
        for chunk in all_results:
            for element in chunk['elements']:
                cls = element['class']
                st.write(f"Element {element['element_index']} ({cls}):")
                if cls == 'Picture':
                    # Display the image
                    if element['result'] is not None:
                        st.image(element['result'], caption=f"Extracted {cls}", use_column_width=True)
                    else:
                        st.write("No image available.")
                elif cls == 'Table':
                    # Display the extracted table (markdown or placeholder)
                    st.write(element['result'])
                else:
                    # Display the extracted text
                    st.write(element['result'])

if __name__ == "__main__":
    main()
