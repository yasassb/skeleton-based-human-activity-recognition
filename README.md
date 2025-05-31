# 🔍 Human Activity Recognition from Video Data

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Ui7Z-_ux)

<div align="center">
  
![Activity Recognition Demo](https://raw.githubusercontent.com/user/repo/main/demo.gif)

*An advanced system for recognizing human activities in video data through skeleton-based analysis*

</div>

## 🌟 Overview

This repository contains a comprehensive solution for human activity recognition in video data using skeletal keypoint extraction and deep learning. The system follows a two-stage approach:

1. **Keypoint Extraction**: Uses MMPOSE to detect and extract human keypoints from video frames
2. **Activity Recognition**: Employs two different models for comparison:
   - CNN-Transformer
   - CNN-LSTM

## 🚀 Features

- Efficient human pose estimation with MMPOSE
- Keypoint normalization for improved recognition
- Two powerful deep learning architectures for comparison
- Interactive GUI for easy model inference


## 🛠️ Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)

- Install MMPOSE to the environment (GitHub: https://github.com/open-mmlab/mmpose)
- Download required MMPOSE model weights:
   ```bash
   # Place these files in the root directory:
   # - td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth
   # - td-hm_hrnet-w48_8xb32-210e_coco-256x192.pth
   ```
- Download the dataset:
   ```bash
   python dataset_downloader.py
   ```

## 💻 Usage

### Running the GUI

```bash
python main.py
```

This launches the interactive GUI where you can:
- Load video files
- Choose between CNN-Transformer or CNN-LSTM models
- Load the model fine and the label encoder file
- View real-time activity recognition results

## 📊 Model Performance

<div align="center">
  
| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| CNN-Transformer | 98.22% | 0.982 | 0.983 | 0.982 |
| CNN-LSTM | 96.59% | 0.966 | 0.966 | 0.965 |

</div>

## 🔄 Pipeline Overview

```mermaid
graph LR
    A[Video Input] --> B[MMPOSE Keypoint Extraction]
    B --> C[Keypoint Normalization]
    C --> D[CNN-Transformer]
    C --> E[CNN-LSTM]
    D --> F[Activity Recognition Results]
    E --> F
```

## 🧰 Tech Stack

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [TensorFlow](https://www.tensorflow.org/) - Machine learning framework
- [MMPOSE](https://github.com/open-mmlab/mmpose) - Human pose estimation toolkit
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [NumPy](https://numpy.org/) - Numerical computing
- [Scikit-learn](https://scikit-learn.org/) - Machine learning utilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

If you have any questions or feedback, please open an issue or contact [yasassb@yahoo.com](mailto:yasassb@yahoo.com) or [sudeeraibm@gmail.com](mailto:sudeeraibm@gmail.com).

---

<div align="center">
  <p>Made with ❤️ for computer vision and deep learning</p>
</div>