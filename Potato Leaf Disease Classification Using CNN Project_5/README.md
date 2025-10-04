# Plant Disease Classification Using CNN  
A Deep Learning Approach on PlantVillage Dataset  

---

## Project Overview
Plant diseases significantly impact crop yield and food security globally. Early detection of plant diseases can help farmers take timely actions, reduce crop loss, and improve productivity.  

In this project, we developed a **Convolutional Neural Network (CNN)** to classify **potato leaf images** into three categories:  

- **Potato___Early_blight**  
- **Potato___Late_blight**  
- **Potato___healthy**  

We used the PlantVillage dataset containing 2,152 images of potato leaves. The workflow includes data preprocessing, dataset splitting, data augmentation, model development, training, evaluation, inference, and saving the trained model for future use.

---

## Problem Statement
The aim of this project is to:  

- Detect early and late blight diseases in potato leaves using images.  
- Build a CNN model that generalizes well on unseen images.  
- Provide farmers or agronomists with a tool to monitor plant health and take preventive actions.

---

## Dataset Details
- **Total Images:** 2,152  
- **Classes:** 3  
  - Potato___Early_blight  
  - Potato___Late_blight  
  - Potato___healthy  
- **Image Size:** 256 x 256 pixels  
- **Channels:** 3 (RGB)  

The dataset was loaded using TensorFlow's `image_dataset_from_directory` function. Images were organized into class-specific directories.

---

## Data Preprocessing
- **Resizing & Normalization:** All images resized to 256x256 and normalized to [0,1] range using a dedicated layer in the CNN.  
- **Data Augmentation:** To increase dataset diversity, we applied horizontal and vertical flipping and random rotation (0.2 radians).  
- **Dataset Splitting:** The dataset was split into training (80%), validation (10%), and test (10%) subsets.  
- **Caching & Prefetching:** For improved performance during training, datasets were cached, shuffled, and prefetched using `tf.data` methods.

---

## Model Architecture
The CNN model architecture includes:  

- **Input Layer** → Resizing & normalization  
- **Convolutional Layers:** 6 Conv2D layers with increasing filters (32 → 64), each followed by MaxPooling2D  
- **Flatten Layer** → Converts 2D feature maps to 1D vector  
- **Dense Layer:** 64 neurons with ReLU activation  
- **Output Layer:** 3 neurons with Softmax activation for multi-class classification  

**Total Trainable Parameters:** 183,747

---

## Model Compilation and Training
- **Optimizer:** Adam  
- **Loss Function:** SparseCategoricalCrossentropy  
- **Metrics:** Accuracy  
- **Batch Size:** 32  
- **Epochs:** 50  

The model was trained with training and validation datasets. Early epochs showed rapid improvement in accuracy, and later epochs demonstrated convergence with minimal validation loss.  

**Final Training Highlights:**  
- **Validation Accuracy:** 100% at the final epoch  
- **Test Accuracy:** 100%  

---

## Evaluation and Visualization
- **Training vs Validation Accuracy:** The model achieved nearly perfect accuracy with low validation loss, indicating excellent learning.  
- **Training vs Validation Loss:** Loss decreased steadily, confirming effective training.  

Visualizations were used to monitor model performance and to validate predictions on sample images.

---

## Inference
A custom `predict` function was implemented for inference on new images:  

predicted_class, confidence = predict(model, image)
The function outputs the predicted class and confidence score for any input image. Sample predictions on the test dataset showed high accuracy and confidence.

---

## Saving the Model
The trained model was saved for future use:
model.save("../potatoes.h5")
This allows deployment or later loading for inference without retraining.

---

## Future Improvements
* Model Enhancements: Explore deeper CNNs, transfer learning (e.g., ResNet, EfficientNet), or attention mechanisms.
* Data Improvements: Include more images from diverse regions, different potato varieties, and varied environmental conditions.
* Real-world Deployment: Develop a web or mobile app for farmers to take a photo of a leaf and instantly get disease prediction.
* Integration with IoT Devices: Use drones or sensors to capture real-time plant images for early detection at scale.

---

## Conclusion

This project successfully developed a CNN-based model for classifying potato leaf diseases with 100% test accuracy. The workflow demonstrates the importance of data preprocessing, data augmentation, and carefully designed CNN architectures in image classification tasks.

With future improvements in data diversity, model complexity, and real-world deployment, this approach can become a valuable decision-support tool for farmers and agronomists, ultimately enhancing crop health monitoring and food security.













predicted_class, confidence = predict(model, image)

