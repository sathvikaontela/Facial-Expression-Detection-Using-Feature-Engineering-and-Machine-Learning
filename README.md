Hi! Welcome to my project on facial expression detection using a combination of traditional feature engineering and deep learning techniques. This project was developed to explore the power of combining different feature extraction methods to classify grayscale facial images.

 🔍 What Was the Goal?
The goal was simple but challenging: accurately classify facial expressions from grayscale images. Raw pixel data alone didn’t perform well, so I focused on transforming these images into more meaningful representations using multiple feature extraction techniques—both classical and modern.

 📦 Dataset and Preprocessing
I worked with a dataset structured into training and testing folders, each containing images of different expression classes. I began by loading all the grayscale images using OpenCV and resized them uniformly to 48x48 pixels. After normalization (scaling pixel values between 0 and 1), the dataset was ready for model building.
To make the model more robust, I also performed data augmentation using Keras’ `ImageDataGenerator`. I introduced variations like rotations, flips, brightness changes, and shifts to help the model generalize better.
🔹 Starting Simple: The Baseline Model
My baseline model used an SVM classifier trained directly on flattened pixel values (raw images). Unsurprisingly, it struggled—accuracy hovered around 34%, and it was heavily biased toward the majority class. This confirmed my hypothesis: we needed strong feature engineering.
 🔬 Feature Engineering Techniques I Used
To improve performance, I extracted features using three different methods:
1. Local Binary Patterns (LBP)
   A lightweight, texture-based method. It compared neighboring pixels to capture local texture patterns, which were summarized into histograms.
2. Histogram of Oriented Gradients (HOG)
   This focused on structural features by capturing gradient orientation distributions. It was especially helpful in understanding shapes within the images.
3. Convolutional Neural Networks (CNN)
   I used a pre-trained VGG16 model (with classification layers removed) to extract deep hierarchical features. I converted grayscale images to RGB for compatibility.
 📉 Reducing Feature Dimensions

All these features produced high-dimensional vectors, which could slow down computation and introduce noise. To address this, I applied PCA (Principal Component Analysis) to reduce each feature type to 50 dimensions. This helped retain essential information while making the data manageable.
 🔗 Bringing It All Together: Feature Fusion
Next, I fused the reduced LBP, HOG, and CNN features by horizontally stacking them into a single combined vector. This fusion gave the model access to a rich mix of texture, shape, and deep semantic information.
 🚀 Final Model and Results
I trained a second SVM model (with RBF kernel) using the fused features. The results showed a significant jump in accuracy to 54.8%. More importantly, performance across multiple classes became balanced. The confusion matrix looked much healthier, and F1-scores improved across the board.

Here’s what stood out:
* Minority classes were better recognized
* Model was less biased toward the majority class
* Classification reports and visualizations confirmed consistent gains

 📊 Visual Insights
To better understand the results, I plotted:
* Confusion Matrices (baseline vs. final model)
* LBP Histograms to explore texture dominance in sample images
These visualizations helped validate that the final model truly leveraged the features meaningfully.

💡 What I Learned
This project reaffirmed how feature engineering can make or break a machine learning model—especially when dealing with images. By combining the best of both worlds—handcrafted features (LBP, HOG) and deep learning (CNNs)—I built a model that was both accurate and computationally efficient.

👨‍💻 Technologies Used
* Python, NumPy, OpenCV
* Scikit-learn (SVM, PCA, metrics)
* TensorFlow/Keras (VGG16, Image Augmentation)
* Scikit-image (LBP, HOG)
* Matplotlib, Seaborn


