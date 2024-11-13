import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64
import json
import fitz  # PyMuPDF for PDF handling
import hashlib  # For content hashing
import os
import gdown
import pytesseract
from table_module import TableExtractor  # Ensure this is defined in table_module.py
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm.auto import tqdm

# Set page config at the very beginning
st.set_page_config(page_title="📄 Document Element Classification and Extraction", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Define constants with more subtle colors
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
    "Unknown": (128, 128, 128)
}
BOX_PADDING = 2

# Define the hierarchy for chunking (not used anymore but kept for potential future use)
SEGMENT_HIERARCHY = [
    "Section-header",
    "Title",
    "Page-header",
    "Page-footer",
    "Table",
    "Picture",
    "Caption",
    "Formula",
    "Text",
    "List-item",
    "Footnote",
    "Unknown"
]

# Flag to control processing of all pages or a single page
PROCESS_ALL_PAGES = True  # Set to True to process all pages, False to process a single page
PAGE_TO_PROCESS = 2       # Specify the page index to process when PROCESS_ALL_PAGES is False (0-based)

# Initialize session state variables
if 'uploaded_files_contents' not in st.session_state:
    st.session_state.uploaded_files_contents = None

if 'images_with_info' not in st.session_state:
    st.session_state.images_with_info = []

if 'detected_elements' not in st.session_state:
    st.session_state.detected_elements = {}

if 'all_results' not in st.session_state:
    st.session_state.all_results = {}

# Initialize TableExtractor
table_extractor = TableExtractor()

# Initialize ThreadPoolExecutor for batch processing
executor = ThreadPoolExecutor(max_workers=4)

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

    line_thickness = max(round(box_scale_factor * (image.shape[0] + image.shape[1]) / 2), 1)

    cv2.rectangle(
        img=image,
        pt1=start_box,
        pt2=end_box,
        color=ENTITIES_COLORS.get(cls, (128, 128, 128)),
        thickness=line_thickness
    )

    text = f"{cls} {detection_class_conf:.2f}"
    font_scale = label_scale_factor
    font_thickness = max(line_thickness - 1, 1)

    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=font_thickness)

    cv2.rectangle(
        image,
        (start_box[0], start_box[1] - text_h - BOX_PADDING * 2),
        (start_box[0] + text_w + BOX_PADDING * 2, start_box[1]),
        ENTITIES_COLORS.get(cls, (128, 128, 128)),
        thickness=-1
    )

    cv2.putText(
        image,
        text,
        (start_box[0] + BOX_PADDING, start_box[1] - BOX_PADDING),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=font_thickness
    )

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

def detect(image, page_numbers=None, page_boundary=None):
    """
    Detect elements in the image and return their details.
    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    preprocessed_image = image_cv

    results = DETECTION_MODEL.predict(source=preprocessed_image, conf=0.2, iou=0.8)
    boxes = results[0].boxes

    detected_elements = []

    if len(boxes) == 0:
        return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)), detected_elements

    for box in boxes:
        detection_class_conf = box.conf.item()
        cls_index = int(box.cls)
        cls = list(ENTITIES_COLORS.keys())[cls_index] if cls_index < len(ENTITIES_COLORS) else "Unknown"

        start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
        end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

        # Determine which page the element belongs to
        if page_boundary is not None and page_numbers is not None:
            if end_box[1] <= page_boundary:
                page_number = page_numbers[0]
            elif start_box[1] >= page_boundary:
                page_number = page_numbers[1]
            else:
                # Element spans across pages
                page_number = page_numbers  # Assign both page numbers
        else:
            page_number = page_numbers[0] if page_numbers else None

        detected_elements.append({
            "class": cls,
            "confidence": detection_class_conf,
            "coordinates": {
                "start": start_box,
                "end": end_box
            },
            "page_number": page_number  # Include page number for continuity detection
        })

        draw_box_and_label(image_cv, start_box, end_box, cls, detection_class_conf)

    # Merge nearby detections
    detected_elements = merge_nearby_detections(detected_elements)

    # Sort detected elements by their position on the page (top to bottom, left to right)
    detected_elements.sort(key=lambda x: (x['coordinates']['start'][1], x['coordinates']['start'][0]))

    # Assign index based on sorted order
    for idx, element in enumerate(detected_elements):
        element['index'] = idx

    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)), detected_elements

def process_pdf(pdf_document):
    """
    Process a PDF document and return a list of combined images from two pages.
    """
    images = []
    
    total_pages = len(pdf_document)
    
    if PROCESS_ALL_PAGES:
        pages_to_process = range(0, total_pages, 2)  # Process two pages at a time
    else:
        # Ensure PAGE_TO_PROCESS is within the valid range
        if PAGE_TO_PROCESS < 0 or PAGE_TO_PROCESS >= total_pages:
            st.error(f"PAGE_TO_PROCESS {PAGE_TO_PROCESS} is out of range. PDF has {total_pages} pages.")
            return images
        pages_to_process = [PAGE_TO_PROCESS]
    
    for i in pages_to_process:
        # Load first page
        page1 = pdf_document.load_page(i)
        pix1 = page1.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
        img1 = Image.frombytes("RGB", [pix1.width, pix1.height], pix1.samples)
        height1 = pix1.height  # Height of the first page image
        
        if i+1 < total_pages:
            # Load second page
            page2 = pdf_document.load_page(i+1)
            pix2 = page2.get_pixmap(matrix=fitz.Matrix(2, 2))
            img2 = Image.frombytes("RGB", [pix2.width, pix2.height], pix2.samples)
            
            # Combine images vertically
            combined_height = img1.height + img2.height
            combined_width = max(img1.width, img2.width)
            combined_image = Image.new('RGB', (combined_width, combined_height))
            combined_image.paste(img1, (0, 0))
            combined_image.paste(img2, (0, img1.height))
            
            images.append((combined_image, [i, i+1], img1.height))  # Include page numbers and boundary
        else:
            images.append((img1, [i], img1.height))

    return images

def crop_image(image, element_region):
    """
    Crop the image to the element region.
    """
    padding = 5  # 5 pixels padding
    x_start = max(element_region["start"][0] - padding, 0)
    y_start = max(element_region["start"][1] - padding, 0)
    x_end = min(element_region["end"][0] + padding, image.width)
    y_end = min(element_region["end"][1] + padding, image.height)

    element_image = image.crop((x_start, y_start, x_end, y_end))

    return element_image  # Return PIL Image object

def process_elements(image, elements):
    """
    Process each detected element individually using OCR or table extraction.
    """
    results = []
    for element in elements:
        cls = element["class"]
        confidence = element["confidence"]
        index = element["index"]
        coordinates = element["coordinates"]

        if cls == "Table":
            # Perform table extraction
            table_image = crop_image(image, coordinates)
            try:
                markdown_table = table_extractor.extract_markdown(table_image)
                # Since TableExtractor returns Markdown, we will use it directly
            except Exception as e:
                markdown_table = f"Error extracting table: {str(e)}"
            results.append({
                "index": index,
                "class": cls,
                "confidence": confidence,
                "markdown": markdown_table,
                "coordinates": coordinates
            })
        elif cls in ["Text", "Section-header", "Title", "Caption", "Footnote", "Page-header", "Page-footer", "List-item"]:
            # Perform OCR
            text_image = crop_image(image, coordinates)
            try:
                text = pytesseract.image_to_string(text_image)
            except Exception as e:
                text = f"Error during OCR: {str(e)}"
            results.append({
                "index": index,
                "class": cls,
                "confidence": confidence,
                "result": text.strip(),
                "coordinates": coordinates
            })
        elif cls == "Picture":
            # Just display the image, no processing
            picture_image = crop_image(image, coordinates)
            buffered = io.BytesIO()
            picture_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_data = f"data:image/png;base64,{img_str}"
            results.append({
                "index": index,
                "class": cls,
                "confidence": confidence,
                "image_data": image_data,
                "coordinates": coordinates
            })
        else:
            # Unknown class, attempt OCR as a fallback
            unknown_image = crop_image(image, coordinates)
            try:
                text = pytesseract.image_to_string(unknown_image)
            except Exception as e:
                text = f"Error during OCR: {str(e)}"
            results.append({
                "index": index,
                "class": cls,
                "confidence": confidence,
                "result": text.strip(),
                "coordinates": coordinates
            })

    results.sort(key=lambda x: x["index"])
    return results

def intelligent_chunking(detected_elements, hierarchy, max_chunk_size=5):
    """
    Group detected elements into chunks based on the given hierarchy.
    """
    chunks = []
    current_chunk_elements = []
    content_hashes = set()

    def hash_content(element):
        # Create a hash of the element's content to check for duplicates
        # Since we don't have the content yet, we'll use coordinates and class
        content = f"{element['class']}_{element['coordinates']}"
        return hashlib.md5(content.encode()).hexdigest()

    for idx, element in enumerate(detected_elements):
        element_type = element['class']
        element_priority = hierarchy.index(element_type) if element_type in hierarchy else len(hierarchy)
        element_hash = hash_content(element)

        # Adjust breaking condition: Only break on high-priority elements
        if element_type == 'Section-header':
            # Start a new chunk
            if current_chunk_elements:
                chunks.append({'elements': current_chunk_elements})
            current_chunk_elements = [element]
            content_hashes = {element_hash}
        else:
            # Add element to current chunk
            if element_hash not in content_hashes:
                current_chunk_elements.append(element)
                content_hashes.add(element_hash)

            # Check if current chunk exceeds max size
            if len(current_chunk_elements) >= max_chunk_size:
                chunks.append({'elements': current_chunk_elements})
                current_chunk_elements = []
                content_hashes = set()

    # Add the last chunk if it's not empty
    if current_chunk_elements:
        chunks.append({'elements': current_chunk_elements})

    # Optional: Merge small chunks with adjacent ones
    merged_chunks = []
    previous_chunk = None
    for chunk in chunks:
        chunk_elements = chunk['elements']
        if previous_chunk and len(previous_chunk['elements']) + len(chunk_elements) <= max_chunk_size:
            previous_chunk['elements'].extend(chunk_elements)
        else:
            if previous_chunk:
                merged_chunks.append(previous_chunk)
            previous_chunk = {'elements': chunk_elements}
    if previous_chunk:
        merged_chunks.append(previous_chunk)

    return merged_chunks

def combine_elements_into_image(image, elements):
    """
    Combine the regions of elements into a single image, and adjust element coordinates.
    """
    # Determine the bounding box that encompasses all elements in the chunk
    x_start = min(element["coordinates"]["start"][0] for element in elements)
    y_start = min(element["coordinates"]["start"][1] for element in elements)
    x_end = max(element["coordinates"]["end"][0] for element in elements)
    y_end = max(element["coordinates"]["end"][1] for element in elements)

    # Crop the image to the bounding box
    combined_image = image.crop((x_start, y_start, x_end, y_end))

    # Adjust the coordinates of the elements
    adjusted_elements = []
    for element in elements:
        adjusted_element = element.copy()
        adjusted_element['coordinates'] = {
            'start': (element['coordinates']['start'][0] - x_start, element['coordinates']['start'][1] - y_start),
            'end': (element['coordinates']['end'][0] - x_start, element['coordinates']['end'][1] - y_start)
        }
        adjusted_elements.append(adjusted_element)

    return combined_image, adjusted_elements

def main():
    st.title("📄 Document Element Classification and Extraction")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload up to 2 scanned document images or a PDF",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        if len(uploaded_files) > 2:
            st.warning("Please upload a maximum of 2 files.")
            st.stop()

        # Read and store the contents of the uploaded files
        uploaded_files_contents = []
        for file in uploaded_files:
            file_content = file.read()
            file.seek(0)  # Reset the file pointer
            uploaded_files_contents.append({
                'name': file.name,
                'type': file.type,
                'content': file_content
            })
        st.session_state.uploaded_files_contents = uploaded_files_contents
        # Clear previous session state data when new files are uploaded
        st.session_state.images_with_info = []
        st.session_state.detected_elements = {}
        st.session_state.all_results = {}
    else:
        if 'uploaded_files_contents' in st.session_state:
            uploaded_files_contents = st.session_state.uploaded_files_contents
        else:
            uploaded_files_contents = []

    if not uploaded_files_contents:
        st.error("Please upload a file to proceed.")
        st.stop()
    else:
        # Process uploaded files
        if not st.session_state.images_with_info:
            images_with_info = []
            for file_info in uploaded_files_contents:
                if file_info['type'] == "application/pdf":
                    # Process PDF
                    pdf_document = fitz.open(stream=file_info['content'], filetype="pdf")
                    images_from_pdf = process_pdf(pdf_document)
                    if not images_from_pdf:
                        st.error("Error processing PDF.")
                        st.stop()
                    images_with_info.extend(images_from_pdf)
                else:
                    # For images, we'll assume each image is a page
                    try:
                        image = Image.open(io.BytesIO(file_info['content'])).convert("RGB")
                        images_with_info.append((image, [0], image.height))
                    except Exception as e:
                        st.error(f"Error opening image {file_info['name']}: {e}")
            st.session_state.images_with_info = images_with_info
        else:
            images_with_info = st.session_state.images_with_info

        for idx, (image, page_numbers, page_boundary) in enumerate(images_with_info):
            st.header(f"Processing Pages {page_numbers}")

            # Create two columns for annotated image and responses
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Annotated Image")
                st.image(image, caption=f"📥 Uploaded Pages {page_numbers}", use_column_width=True)

            with col2:
                st.subheader("Detected Elements and Extraction")
                detect_button_key = f"detect_{idx}"
                if st.button(f"🔍 Detect Elements for Pages {page_numbers}", key=detect_button_key):
                    with st.spinner(f"Processing Pages {page_numbers}..."):
                        result_image, detected_elements = detect(image, page_numbers, page_boundary)
                        # Update the annotated image in col1
                        col1.image(result_image, caption=f"✅ Detected Elements in Pages {page_numbers}", use_column_width=True)

                        if detected_elements:
                            st.write(f"Detected {len(detected_elements)} element(s) in Pages {page_numbers}. Processing...")

                            # Store detected elements
                            st.session_state.detected_elements[idx] = detected_elements

                            # Submit processing to the ThreadPoolExecutor
                            future = executor.submit(process_elements, image, detected_elements)

                            # Asynchronously display results as they complete
                            with st.spinner("Extracting information from detected elements..."):
                                try:
                                    chunk_results = future.result(timeout=300)  # Set timeout as needed
                                except Exception as e:
                                    st.error(f"Error during processing: {e}")
                                    continue

                            # Store the results
                            st.session_state.all_results[idx] = chunk_results

                            # Save all results to response.json
                            try:
                                with open("response.json", "w") as f:
                                    json.dump(st.session_state.all_results, f, indent=4)
                                st.success("Processing complete. Results saved to response.json.")
                            except Exception as e:
                                st.error(f"Error writing response file: {e}")

                            # Display results for each element
                            st.markdown(f"#### Extracted Information for Pages {page_numbers}:")
                            for result in chunk_results:
                                st.markdown(f"**Element {result['index']} ({result['class']}):**")
                                if result['class'] == "Table":
                                    st.markdown("**Extracted Table (Markdown):**")
                                    st.markdown(result.get('markdown', ''))
                                elif result['class'] in ["Text", "Section-header", "Title", "Caption", "Footnote", "Page-header", "Page-footer", "List-item"]:
                                    st.markdown("**Extracted Text:**")
                                    st.write(result.get('result', ''))
                                elif result['class'] == "Picture":
                                    st.markdown("**Picture:**")
                                    st.image(result.get('image_data', ''), use_column_width=True)
                                else:
                                    st.markdown("**Extracted Content:**")
                                    st.write(result.get('result', ''))
                        else:
                            st.write(f"No elements detected in Pages {page_numbers}.")
                else:
                    # Check if detected elements and results are in session state
                    if idx in st.session_state.all_results:
                        chunk_results = st.session_state.all_results[idx]
                        st.markdown(f"#### Extracted Information for Pages {page_numbers}:")
                        for result in chunk_results:
                            st.markdown(f"**Element {result['index']} ({result['class']}):**")
                            if result['class'] == "Table":
                                st.markdown("**Extracted Table (Markdown):**")
                                st.markdown(result.get('markdown', ''))
                            elif result['class'] in ["Text", "Section-header", "Title", "Caption", "Footnote", "Page-header", "Page-footer", "List-item"]:
                                st.markdown("**Extracted Text:**")
                                st.write(result.get('result', ''))
                            elif result['class'] == "Picture":
                                st.markdown("**Picture:**")
                                st.image(result.get('image_data', ''), use_column_width=True)
                            else:
                                st.markdown("**Extracted Content:**")
                                st.write(result.get('result', ''))
                    else:
                        st.info("Click the button above to detect elements and extract information.")

if __name__ == "__main__":
    main()
