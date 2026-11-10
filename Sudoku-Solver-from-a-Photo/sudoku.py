#!/usr/bin/env python
# coding: utf-8

# # Arshia Goshtasbi
# ## Machine Learning, Capstone Project: Sudoku Solver from a Photo (Phases 1-4)

# # Imports

# In[ ]:


import os
import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[ ]:


# Paths (relative to this folder). Originally run on Google Colab.
DATA_DIR  = os.path.join('assets', 'assets')   # printed-digits dataset, unzipped (see README)
MODEL_DIR = 'models'
TEST_DIR  = 'Test'      # put your own Sudoku photos here
RESULT_DIR = 'Result'


# ### Utility Function

# In[ ]:


def display_image(image, title='Image'):
    plt.figure(figsize=(8, 8))
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# ## Phase 1: Paper Deskewing & Rectification

# In[ ]:


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)


def find_paper_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_corners(paper_contour):
    epsilon = 0.02 * cv2.arcLength(paper_contour, True)
    approx = cv2.approxPolyDP(paper_contour, epsilon, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(paper_contour)
        approx = cv2.boxPoints(rect)
    corners = order_points(approx.reshape(4, 2))
    return np.array(corners, dtype=np.float32)


def perspective_transform(image, corners):
    (tl, tr, br, bl) = corners
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def phase1_rectify(input_path, save_path=None, visualize=True):
    image = cv2.imread(input_path)
    display_image(image, "Original")
    if image is None:
        raise FileNotFoundError(f"Cannot read {input_path}")
    orig = image.copy()

    edges = preprocess_image(image)
    if visualize: display_image(edges, "Edges for Paper Detection")

    paper = find_paper_contour(edges)
    outline = orig.copy()
    if paper is not None:
        cv2.drawContours(outline, [paper], -1, (0, 255, 0), 2)
    if visualize: display_image(outline, "Detected Paper Contour")
    if paper is None:
        raise RuntimeError("Paper contour not found.")

    corners = find_corners(paper)
    warped = perspective_transform(image, corners)
    if visualize: display_image(warped, "Phase 1: Rectified Paper")

    if save_path:
        cv2.imwrite(save_path, warped)
    return warped


# ## Phase 2: Sudoku Grid Detection & Drawing

# In[ ]:


def find_sudoku_grid(rectified):
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            if 0.8 <= ar <= 1.2:
                return approx.reshape(4, 2), thresh
    return None, thresh

def phase2_draw_grid(rectified, save_path=None, visualize=True):
    overlay = rectified.copy()
    grid_cnt, thresh = find_sudoku_grid(rectified)
    if grid_cnt is not None:
        cv2.drawContours(overlay, [grid_cnt], -1, (0, 255, 0), 2)
    if visualize: display_image(overlay, 'Phase 2: Grid Overlay')
    if save_path:
        cv2.imwrite(save_path, overlay)
    return overlay


# ## Main Execution (Just One Test Case)

# In[ ]:


image_path = "Test/synthetic_1.jpg"                                    # Path to your input Sudoku image
phase1_output_path = f"phase1_rectified_paper.jpg"               # Desired filename for Phase 1 result
phase2_output_path = f"phase2_grid_detection.jpg"                # Desired filename for Phase 2 result

# Run Phase 1
print("Running Phase 1: Deskew & Rectify Paper...")
rectified = phase1_rectify(image_path, save_path=phase1_output_path, visualize=True)
print(f"Phase 1 output saved to {phase1_output_path}")

# Run Phase 2
print("Running Phase 2: Detect & Draw Sudoku Grid...")
overlay = phase2_draw_grid(rectified, save_path=phase2_output_path, visualize=True)
print(f"Phase 2 output saved to {phase2_output_path}")

print("Done. All results saved.")


# # Batch Processing

# In[ ]:


os.makedirs("Result", exist_ok=True)

# Process all images in Test folder
test_folder = "Test"
for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"\nProcessing {filename}...")
        image_path = os.path.join(test_folder, filename)

        try:
            base_name = os.path.splitext(filename)[0]
            phase1_output = os.path.join("Result", f"{base_name}_phase1.jpg")
            phase2_output = os.path.join("Result", f"{base_name}_phase2.jpg")

            # Run Phase 1
            rectified = phase1_rectify(image_path, save_path=phase1_output, visualize=True)

            # Run Phase 2
            drawn_grid = phase2_draw_grid(rectified, save_path=phase2_output, visualize=True)

            print(f"Results saved for {filename}:")
            print(f" - {phase1_output}")
            print(f" - {phase2_output}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("\nBatch processing completed!")


# ## Phase 3: Cell Segmentation and Digit Recognition

# ## Model Training
# Dataset is assets.zip file that is from kaggle:
# https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset

# In[ ]:


# Define paths
zip_path = 'assets.zip'
extract_path = 'assets'
# Unzip the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


# # Simple Model

# In[ ]:


# === Parameters ===
batch_size  = 32
img_height  = 28
img_width   = 28
data_dir    = DATA_DIR  # root folder containing 10 subfolders (0,1,...,9)
epochs      = 10

# === Load datasets ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'            # ensures labels are integers 0–9
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

# === Performance optimizations ===
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache() \
                   .shuffle(1000) \
                   .map(lambda x, y: (x/255.0, y),
                        num_parallel_calls=AUTOTUNE) \
                   .prefetch(AUTOTUNE)

val_ds   = val_ds.cache() \
                 .map(lambda x, y: (x/255.0, y),
                      num_parallel_calls=AUTOTUNE) \
                 .prefetch(AUTOTUNE)

# === Build the model ===
model = models.Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')      # 10 output neurons for digits 0–9
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Train ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# === Save ===
model.save(
    os.path.join(MODEL_DIR, "digit_classifier.h5"),
    include_optimizer=False
)

# === (Optional) Plot training history ===
import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Loss")
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Accuracy")
plt.show()


# # Advanced Model
# -> Data Augmentation
# 
# -> Better Preproccess
# 
# -> Complex Model
# 

# In[ ]:


# === Parameters ===
batch_size  = 32
img_height  = 28
img_width   = 28
data_dir    = DATA_DIR
epochs      = 50  # Increased epochs with early stopping

# === Load datasets ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

# === Data Augmentation ===
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),  # Rotate up to 10 degrees
    layers.RandomZoom(0.1),      # Zoom in/out by 10%
    layers.RandomTranslation(0.1, 0.1),  # Shift by 10%
    layers.RandomContrast(0.2),  # Adjust contrast
])

# === Enhanced preprocessing ===
def preprocess_data(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

def augment_data(x, y):
    x = data_augmentation(x, training=True)
    return x, y

AUTOTUNE = tf.data.AUTOTUNE

# Apply augmentation only to training data
train_ds = train_ds.cache() \
                   .map(preprocess_data, num_parallel_calls=AUTOTUNE) \
                   .map(augment_data, num_parallel_calls=AUTOTUNE) \
                   .shuffle(2000) \
                   .prefetch(AUTOTUNE)

val_ds = val_ds.cache() \
               .map(preprocess_data, num_parallel_calls=AUTOTUNE) \
               .prefetch(AUTOTUNE)

# === Enhanced Model Architecture ===
model = models.Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 1)),

    # First Conv Block
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),

    # Second Conv Block
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),

    # Third Conv Block
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.GlobalAveragePooling2D(),  # Better than Flatten
    layers.Dropout(0.5),

    # Dense layers
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# === Advanced Compilation ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks for better training ===
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# === Train with callbacks ===
print("Model Summary:")
model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# === Evaluate and analyze results ===
test_loss, test_accuracy = model.evaluate(val_ds, verbose=0)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

# === Save the improved model ===
model.save(
    os.path.join(MODEL_DIR, "improved_digit_classifier.h5"),
    include_optimizer=False
)

# === Detailed analysis for digit "1" ===
def analyze_digit_performance(model, val_ds):
    """Analyze per-class performance to identify issues with specific digits"""
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Specific analysis for digit "1"
    digit_1_indices = [i for i, label in enumerate(y_true) if label == 1]
    digit_1_predictions = [y_pred[i] for i in digit_1_indices]
    digit_1_accuracy = sum(1 for pred in digit_1_predictions if pred == 1) / len(digit_1_predictions)

    print(f"\nDigit '1' specific accuracy: {digit_1_accuracy:.4f}")
    print(f"Common misclassifications for '1': {set(digit_1_predictions) - {1}}")

# Run analysis
analyze_digit_performance(model, val_ds)

# === Enhanced plotting ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_title('Model Loss Over Time')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax2.set_title('Model Accuracy Over Time')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Function to test individual predictions ===
def test_prediction(model, image_path):
    """Test the model on a single image"""
    img = tf.keras.utils.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.bar(range(10), predictions[0])
    plt.title(f'Predicted: {predicted_class} (Confidence: {confidence:.3f})')
    plt.xlabel('Digit Class')
    plt.ylabel('Probability')
    plt.xticks(range(10))
    plt.show()

    return predicted_class, confidence


# # Recognition

# In[ ]:


def detect_cell_coords(rectified_image):
    """Detect Sudoku grid cells using contour analysis"""
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find largest contour (grid)
    grid_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(grid_contour)

    # Calculate cell dimensions
    cell_width = w // 9
    cell_height = h // 9

    # Generate cell coordinates (with padding to avoid grid lines)
    cell_coords = []
    for i in range(9):
        row = []
        for j in range(9):
            x0 = x + j * cell_width + 2 # +2px padding
            y0 = y + i * cell_height + 2
            x1 = x0 + cell_width - 4     # -4px total padding
            y1 = y0 + cell_height - 4
            row.append(((x0, y0), (x1, y1)))
        cell_coords.append(row)

    return cell_coords

def extract_digit(cell_img, threshold=0.05):
    """
    Extract and center the digit from a Sudoku cell
    Returns a 28x28 image with centered digit and removed grid lines
    """
    # Apply aggressive thresholding to remove any gray areas
    _, thresh = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY)

    # Find contours to locate the digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return blank image
    if not contours:
        return np.zeros((28, 28), dtype=np.uint8)

    # Find the largest contour (should be the digit)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Create a mask to isolate the digit
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    # Apply the mask to get only the digit
    digit_only = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Crop to the digit region with 2px padding
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(cell_img.shape[1], x + w + padding)
    y2 = min(cell_img.shape[0], y + h + padding)
    digit_crop = digit_only[y1:y2, x1:x2]

    # Skip if the digit is too small
    if digit_crop.size == 0 or np.max(digit_crop) < 50:
        return np.zeros((28, 28), dtype=np.uint8)

    # Resize the digit to 20x20 while maintaining aspect ratio
    h, w = digit_crop.shape
    if h > w:
        new_h = 25
        new_w = int(25 * w / h)
    else:
        new_w = 25
        new_h = int(25 * h / w)
    resized = cv2.resize(digit_crop, (new_w, new_h))

    # Create a 28x28 canvas and place the digit in the center
    canvas = np.zeros((28, 28), dtype=np.uint8)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    return canvas

def segment_and_recognize(rectified, model, cell_coords, visualize=True):
    """Segment grid and recognize digits with enhanced preprocessing"""
    grid = [[0]*9 for _ in range(9)]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    for i in range(9):
        for j in range(9):
            (x0, y0), (x1, y1) = cell_coords[i][j]
            cell_img = gray[y0:y1, x0:x1]

            # Skip empty cells (mostly white)
            if np.mean(cell_img) > 240:
                grid[i][j] = 0
                if visualize:
                    plt.figure(figsize=(1.5, 1.5))
                    plt.imshow(cell_img, cmap='gray')
                    plt.title(f"({i},{j}) → Empty")
                    plt.axis('off')
                    plt.show()
                continue

            # Extract and center the digit
            digit_img = extract_digit(cell_img)

            # Prepare for prediction
            processed = digit_img.astype('float32') / 255.0
            processed = processed.reshape(1, 28, 28, 1)

            # Predict digit
            pred = model.predict(processed, verbose=0)
            digit = np.argmax(pred)
            confidence = np.max(pred)

            if confidence > 0.8:  # Only accept confident predictions
                grid[i][j] = digit
            else:
                grid[i][j] = 0

            if visualize:
                # Show original cell and processed digit
                fig, axs = plt.subplots(1, 2, figsize=(3, 1.5))
                axs[0].imshow(cell_img, cmap='gray')
                axs[0].set_title(f"Original Cell ({i},{j})")
                axs[0].axis('off')

                axs[1].imshow(digit_img, cmap='gray')
                axs[1].set_title(f"Digit: {digit}")
                axs[1].axis('off')

                plt.tight_layout()
                plt.show()

    return grid


# # Phase 4
# Solve the Puzzle using Backtracing

# In[ ]:


def solve_sudoku(grid):
    """Sudoku solving algorithm using backtracking"""
    def is_valid(board, row, col, num):
        # Check row
        if num in board[row]:
            return False

        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False

        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False
        return True

    def backtrack(board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack(board):
                                return True
                            board[i][j] = 0
                    return False
        return True

    # Make a copy to avoid modifying original
    board_copy = [row[:] for row in grid]
    if backtrack(board_copy):
        return board_copy
    return None

def draw_solution(image, cell_coords, original_grid, solved_grid):
    """Draw solution on the Sudoku grid image"""
    result = image.copy()
    for i in range(9):
        for j in range(9):
            if original_grid[i][j] == 0:  # Only draw solutions for empty cells
                (x0, y0), (x1, y1) = cell_coords[i][j]
                center_x = (x0 + x1) // 2
                center_y = (y0 + y1) // 2

                cv2.putText(result, str(solved_grid[i][j]),
                           (center_x - 10, center_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return result


# # Main Execution ALL 4 Phases

# In[ ]:


def main_pipeline(image_path, model_path, visualize=True):
    """Complete pipeline from image to solved Sudoku"""
    # Phase 1: Deskew and rectify
    rectified = phase1_rectify(image_path, visualize=visualize)

    # Phase 2: Detect grid cells
    cell_coords = detect_cell_coords(rectified)
    if cell_coords is None:
        # Fallback: use entire image as grid
        h, w = rectified.shape[:2]
        cell_w, cell_h = w//9, h//9
        cell_coords = [[((j*cell_w, i*cell_h), ((j+1)*cell_w, (i+1)*cell_h))
                      for j in range(9)] for i in range(9)]

    # Phase 3: Recognize digits
    model = load_model(model_path, compile=False)
    unsolved_grid = segment_and_recognize(rectified, model, cell_coords, visualize=visualize)

    print("\nRecognized Sudoku:")
    for row in unsolved_grid:
        print(row)

    # Phase 4: Solve Sudoku
    solved_grid = solve_sudoku(unsolved_grid)
    if solved_grid:
        print("\nSolved Sudoku:")
        for row in solved_grid:
            print(row)

        # Draw solution on image
        solution_image = draw_solution(rectified, cell_coords, unsolved_grid, solved_grid)
        if visualize: display_image(solution_image, "Solved Sudoku")

        return solution_image, unsolved_grid, solved_grid
    else:
        print("Could not solve this Sudoku puzzle")
        return rectified, unsolved_grid, None


# In[ ]:


# Main execution
if __name__ == "__main__":
    # Configuration
    image_path = "Test/synthetic_3.jpg"  # Replace with your Sudoku image path
    model_path = os.path.join(MODEL_DIR, "improved_digit_classifier.h5")
    # model_path = os.path.join(MODEL_DIR, "digit_classifier.h5")

    # Run complete pipeline
    solution_image, unsolved_grid, solved_grid = main_pipeline(image_path, model_path, visualize=True)

    # Save results
    if solved_grid:
        cv2.imwrite("solved_sudoku.jpg", solution_image)
        print("Solved image saved as solved_sudoku.jpg")

