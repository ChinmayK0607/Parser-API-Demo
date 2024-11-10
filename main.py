import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64
from openai import OpenAI
import json
import fitz  # PyMuPDF for PDF handling
import hashlib  # For content hashing
import os
import gdown
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter

# Set page config at the very beginning
st.set_page_config(page_title="Document Element Classification", layout="wide")

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

# Define the hierarchy for chunking
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

# Initialize OpenAI API key in session state
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# Function to load YOLO model with caching to improve performance
MODEL_PATH = "models/yolov10x_best.pt"
FILE_ID = "1jTF4xd0Pu7FDFpLTfSGjgTTolZju4_j7"  # Extracted from your shareable link
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading YOLO model from Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            # Download the file using gdown
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
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

def crop_and_encode_image(image, element_region):
    """
    Crop the image to the element region and encode it as base64.
    """
    element_image = image.crop((
        element_region["coordinates"]["start"][0],
        element_region["coordinates"]["start"][1],
        element_region["coordinates"]["end"][0],
        element_region["coordinates"]["end"][1]
    ))

    # Maintain aspect ratio while resizing
    max_size = (800, 800)  # Reduced size to manage memory usage
    element_image.thumbnail(max_size, Image.LANCZOS)

    buffered = io.BytesIO()
    element_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str

def process_elements(image, elements, openai_api_key):
    """
    Process each detected element individually.
    """
    results = []

    prompts = {
        "Text": "Extract the text from the provided image.",
        "Section-header": "Extract the text from the provided image.",
        "Title": "Extract the text from the provided image.",
        "Caption": "Extract the text from the provided image.",
        "Footnote": "Extract the text from the provided image.",
        "Page-header": "Extract the text from the provided image.",
        "Page-footer": "Extract the text from the provided image.",
        "List-item": "Extract the text from the provided image.",
        "Table": "Generate the HTML code for the table in the provided image and provide a summary of the data in the table. The HTML should be semantically correct and use appropriate tags like <table>, <tr>, <th>, and <td>. Please output the results separated by '---'. First provide the HTML code, then a summary of the data.",
        "Picture": "Provide a detailed description of the provided image. Extract any text/tables present in the image and give its html or markdown format, whichever is more suitable for you.Make sure that you explain in detail any trend that you see in the picture.",
        "Formula": "Simplify the mathematical formula shown in the provided image. Give its latex formula if possible.",
        "Unknown": "Describe the content of the provided image."
    }

    # Initialize OpenAI client with the provided API key
    KEY = openai_api_key
    client = OpenAI(api_key=KEY)
    for element in elements:
        cls = element["class"]
        prompt = prompts.get(cls, "Describe the content of the provided image.")
        auxiliary_prompt = "IN CASE YOU CANNOT DETECT OR HELP WITH THE EXTRACTION, ADD AN <UNK> TOKEN SIMPLY. NOTHING ELSE. Do not add placeholder or introductory or explainatory text like <sure heres your text> or <the text in the image is as follows> or basically anything that is outside of the task asked for. Just perform the instructed task. Response should be limited to the things that you are asked for, not anything more, nor anything less. Output must be provided in markdown format only."
        prompt += auxiliary_prompt
        img_str = crop_and_encode_image(image, element)
        image_data = f"data:image/png;base64,{img_str}"

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data,
                    "detail": "auto"
                },
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=3000,
            )

            full_result = response.choices[0].message.content.strip()
        except Exception as e:
            full_result = f"Error processing element: {str(e)}"

        if cls == "Table":
            individual_results = full_result.split('---')
            if len(individual_results) >= 2:
                html_code = individual_results[0].strip()
                summary = individual_results[1].strip()
            else:
                html_code = full_result.strip()
                summary = ""
            results.append({
                "index": element["index"],
                "class": cls,
                "confidence": element["confidence"],
                "html": html_code,
                "summary": summary,
                "coordinates": element["coordinates"]
            })
        else:
            results.append({
                "index": element["index"],
                "class": cls,
                "confidence": element["confidence"],
                "result": full_result,
                "coordinates": element["coordinates"]
            })

    results.sort(key=lambda x: x["index"])
    return results

# Implement the improved chunking algorithm
def improved_intelligent_chunking_with_continuity(detected_elements, hierarchy, max_chunk_size=5):
    """
    Group detected elements into larger chunks, considering content continuity.

    Parameters:
    - detected_elements: List of detected elements sorted by their position.
    - hierarchy: List defining the priority of segment types.
    - max_chunk_size: Maximum number of elements per chunk.

    Returns:
    - List of chunks, where each chunk is a dictionary containing elements and images.
    """
    chunks = []
    current_chunk_elements = []
    content_hashes = set()
    element_count = len(detected_elements)

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

def determine_dominant_class(classes):
    """
    Determine the dominant class in a list of classes based on hierarchy.
    """
    for cls in SEGMENT_HIERARCHY:
        if cls in classes:
            return cls
    return "Unknown"

def encode_image(image):
    """
    Encode PIL image to base64 string.
    """
    # Maintain aspect ratio while resizing
    max_size = (800, 800)
    image.thumbnail(max_size, Image.LANCZOS)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str

def get_prompt_for_class(cls):
    """
    Get the prompt for a given class.
    """
    prompts = {
        "Text": "Extract the text from the provided image.",
        "Section-header": "Extract the text from the provided image.",
        "Title": "Extract the text from the provided image.",
        "Caption": "Extract the text from the provided image.",
        "Footnote": "Extract the text from the provided image.",
        "Page-header": "Extract the text from the provided image.",
        "Page-footer": "Extract the text from the provided image.",
        "List-item": "Extract the text from the provided image.",
        "Table": "Generate the HTML code for the table in the provided image and provide a summary of the data in the table. The HTML should be semantically correct and use appropriate tags like <table>, <tr>, <th>, and <td>. Please output the results separated by '---'. First provide the HTML code, then a summary of the data.",
        "Picture": "Provide a detailed description of the provided image. Extract any text/tables present in the image and give its html or markdown format, whichever is more suitable for you.",
        "Formula": "Simplify the mathematical formula shown in the provided image. Give its latex formula if possible.",
        "Unknown": "Describe the content of the provided image."
    }
    return prompts.get(cls, "Describe the content of the provided image.")

def main():
    st.title("📄 Document Element Classification with Extraction")

    # Step 1: Prompt user for OpenAI API Key if not already provided
    if not st.session_state.openai_api_key:
        st.header("🔑 Enter Your OpenAI API Key")
        api_key_input = st.text_input(
            "Please enter your OpenAI API key:",
            type="password",
            placeholder="sk-****************************",
            help="You can obtain your API key from https://platform.openai.com/account/api-keys"
        )

        if st.button("Submit API Key"):
            if api_key_input.strip() == "":
                st.error("API key cannot be empty. Please enter a valid OpenAI API key.")
            else:
                st.session_state.openai_api_key = api_key_input.strip()
                st.success("API key successfully saved!")
                # No rerun, the script will naturally continue rendering below

    # Only proceed if API key is present
    if st.session_state.openai_api_key:
        OPENAI_API_KEY = st.session_state.openai_api_key
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Initialize session state variables
        if 'uploaded_files_contents' not in st.session_state:
            st.session_state.uploaded_files_contents = None

        if 'images_with_info' not in st.session_state:
            st.session_state.images_with_info = []

        if 'detected_elements' not in st.session_state:
            st.session_state.detected_elements = {}

        if 'chunks' not in st.session_state:
            st.session_state.chunks = {}

        if 'all_results' not in st.session_state:
            st.session_state.all_results = {}

        if 'total_chunks' not in st.session_state:
            st.session_state.total_chunks = 0

        if 'processed_chunks' not in st.session_state:
            st.session_state.processed_chunks = 0

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload up to 2 scanned document images or a PDF",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        if uploaded_files:
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
            st.session_state.chunks = {}
            st.session_state.all_results = {}
            st.session_state.total_chunks = 0
            st.session_state.processed_chunks = 0
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
                            st.stop()
                        images_with_info.extend(images_from_pdf)
                    else:
                        # For images, we'll assume each image is a page
                        image = Image.open(io.BytesIO(file_info['content']))
                        images_with_info.append((image, [0], image.height))
                st.session_state.images_with_info = images_with_info
            else:
                images_with_info = st.session_state.images_with_info

            # Calculate total number of chunks
            if st.session_state.total_chunks == 0:
                total_chunks = 0
                for idx, (image, page_numbers, page_boundary) in enumerate(images_with_info):
                    result_image, detected_elements = detect(image, page_numbers, page_boundary)
                    if detected_elements:
                        chunks = improved_intelligent_chunking_with_continuity(detected_elements, SEGMENT_HIERARCHY, max_chunk_size=10)
                        total_chunks += len(chunks)
                st.session_state.total_chunks = total_chunks

            for idx, (image, page_numbers, page_boundary) in enumerate(images_with_info):
                st.header(f"Processing Pages {page_numbers}")

                # Create two columns for annotated image and responses
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("Annotated Image")
                    st.image(image, caption=f"📥 Uploaded Pages {page_numbers}", use_column_width=True)

                with col2:
                    st.subheader("Detected Elements and Chunks")
                    detect_button_key = f"detect_{idx}"
                    if st.button(f"🔍 Detect Elements and Create Chunks for Pages {page_numbers}", key=detect_button_key):
                        with st.spinner(f"Processing Pages {page_numbers}..."):
                            result_image, detected_elements = detect(image, page_numbers, page_boundary)
                            # Update the annotated image in col1
                            col1.image(result_image, caption=f"✅ Detected Elements in Pages {page_numbers}", use_column_width=True)

                            if detected_elements:
                                st.write(f"Detected {len(detected_elements)} element(s) in Pages {page_numbers}. Creating chunks...")

                                # Apply the chunking algorithm
                                chunks = improved_intelligent_chunking_with_continuity(detected_elements, SEGMENT_HIERARCHY, max_chunk_size=10)
                                st.write(f"Generated {len(chunks)} chunk(s) from detected elements.")

                                # For each chunk, generate the images and store them
                                for chunk_idx, chunk in enumerate(chunks):
                                    # Combine elements into an image
                                    chunk_image, adjusted_elements = combine_elements_into_image(image, chunk['elements'])
                                    # Run detection on the chunk image to get the annotated image
                                    annotated_chunk_image, _ = detect(chunk_image)

                                    # Store images and chunk index in the chunk data
                                    chunk['chunk_index'] = chunk_idx
                                    chunk['original_image'] = chunk_image
                                    chunk['annotated_image'] = annotated_chunk_image
                                    # Update elements with adjusted coordinates
                                    chunk['elements'] = adjusted_elements

                                # Store detected elements and chunks in session state
                                st.session_state.detected_elements[idx] = detected_elements
                                st.session_state.chunks[idx] = chunks

                                # Display chunks
                                display_chunks(idx, chunks, page_numbers)
                            else:
                                st.write(f"No elements detected in Pages {page_numbers}.")
                    else:
                        # Check if detected elements and chunks are in session state
                        if idx in st.session_state.chunks:
                            chunks = st.session_state.chunks[idx]
                            # Display chunks
                            display_chunks(idx, chunks, page_numbers)
                        else:
                            st.info("Please click the button above to detect elements and create chunks.")

            # Check if all chunks have been processed
            if st.session_state.processed_chunks == st.session_state.total_chunks and st.session_state.total_chunks > 0:
                # Generate JSON data
                json_data = json.dumps(st.session_state.all_results, indent=4)
                # Encode JSON data to base64
                b64_json = base64.b64encode(json_data.encode()).decode()

                # Generate PDF
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()

                # Define a 'Code' style only if it doesn't already exist
                if 'Code' not in styles.byName:
                    styles.add(ParagraphStyle(name='Code', parent=styles['Normal'], fontName='Courier', fontSize=8))

                elements = []
                # For each result in all_results
                for idx, page_results in st.session_state.all_results.items():
                    for chunk_result in page_results:
                        pages = chunk_result["pages"]
                        elements.append(Paragraph(f"Pages: {pages}", styles['Heading2']))
                        elements.append(Spacer(1, 12))
                        # Include annotated chunk image
                        annotated_image = st.session_state.chunks[idx][chunk_result["chunk_index"]]['annotated_image']
                        img_buffer = io.BytesIO()
                        annotated_image.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        rl_image = RLImage(img_buffer, width=400)  # Adjust width as needed
                        elements.append(rl_image)
                        elements.append(Spacer(1, 12))
                        for result in chunk_result['chunk_results']:
                            elements.append(Paragraph(f"Element {result['index']} ({result['class']}):", styles['Heading3']))
                            if result['class'] == "Table":
                                elements.append(Paragraph("HTML Code:", styles['Normal']))
                                elements.append(Preformatted(result.get('html', ''), styles['Code']))
                                elements.append(Paragraph("Summary:", styles['Normal']))
                                elements.append(Paragraph(result.get('summary', ''), styles['Normal']))
                            else:
                                elements.append(Paragraph(result.get('result', ''), styles['Normal']))
                            elements.append(Spacer(1, 12))
                        elements.append(PageBreak())
                # Build the PDF
                doc.build(elements)
                buffer.seek(0)
                # Encode PDF data to base64
                b64_pdf = base64.b64encode(buffer.read()).decode()

                # JavaScript to trigger download
                download_js = f"""
                    <script>
                        function downloadBase64File(contentBase64, contentType, fileName) {{
                            const linkSource = `data:${{contentType}};base64,${{contentBase64}}`;
                            const downloadLink = document.createElement("a");
                            downloadLink.href = linkSource;
                            downloadLink.download = fileName;
                            downloadLink.click();
                        }}
                        // Download JSON
                        downloadBase64File("{b64_json}", "application/json", "all_responses.json");
                        // Download PDF
                        downloadBase64File("{b64_pdf}", "application/pdf", "all_responses.pdf");
                    </script>
                """
                # Display a message
                st.success("All chunks processed. Your downloads will start shortly.")
                # Inject the JavaScript into the page
                st.markdown(download_js, unsafe_allow_html=True)

def display_chunks(idx, chunks, page_numbers):
    """
    Helper function to display chunks and process them.
    """
    for chunk_idx, chunk in enumerate(chunks):
        st.markdown(f"### Chunk {chunk_idx+1}")
        chunk_classes = [element['class'] for element in chunk['elements']]
        st.write(f"**Classes in chunk:** {chunk_classes}")
        for element in chunk['elements']:
            st.write(f"- Element {element['index']}: {element['class']} on page {element['page_number']}")
        # Display chunk images using expander
        with st.expander(f"Show Chunk {chunk_idx+1} Images"):
            # Display the original chunk image
            st.image(chunk['original_image'], caption=f"Original Chunk {chunk_idx+1} Image", use_column_width=True)
            # Display the annotated chunk image
            st.image(chunk['annotated_image'], caption=f"Annotated Chunk {chunk_idx+1} Image", use_column_width=True)

        # Check if result already exists
        if idx in st.session_state.all_results and len(st.session_state.all_results[idx]) > chunk_idx:
            chunk_results = st.session_state.all_results[idx][chunk_idx]["chunk_results"]
            st.markdown(f"#### Results for Chunk {chunk_idx+1}:")
            for result in chunk_results:
                st.markdown(f"**Element {result['index']} ({result['class']}):**")
                if result['class'] == "Table":
                    st.markdown("**HTML Code:**")
                    st.code(result.get('html', ''), language='html')
                    st.markdown("**Summary:**")
                    st.write(result.get('summary', ''))
                else:
                    st.write(result.get('result', ''))
        else:
            # Process the chunk's elements
            with st.spinner(f"Processing Chunk {chunk_idx+1}..."):
                # Process elements individually
                chunk_results = process_elements(chunk['original_image'], chunk['elements'], st.session_state.openai_api_key)

                # Store the results
                if idx not in st.session_state.all_results:
                    st.session_state.all_results[idx] = []
                st.session_state.all_results[idx].append({
                    "pages": page_numbers,
                    "chunk_index": chunk_idx,
                    "chunk_results": chunk_results
                    # Remove 'annotated_image' from all_results to prevent serialization errors
                })

                # Update processed chunks count
                st.session_state.processed_chunks += 1

                # Display results for each element
                st.markdown(f"#### Results for Chunk {chunk_idx+1}:")
                for result in chunk_results:
                    st.markdown(f"**Element {result['index']} ({result['class']}):**")
                    if result['class'] == "Table":
                        st.markdown("**HTML Code:**")
                        st.code(result.get('html', ''), language='html')
                        st.markdown("**Summary:**")
                        st.write(result.get('summary', ''))
                    else:
                        st.write(result.get('result', ''))

if __name__ == "__main__":
    main()
