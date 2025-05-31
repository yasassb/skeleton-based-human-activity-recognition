import sys
import os
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QFileDialog, QLabel, QCheckBox,
                             QDoubleSpinBox, QComboBox, QLineEdit, QTabWidget,
                             QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QProcess

class ModelInferenceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Activity Recognition Inference")
        self.setGeometry(100, 100, 800, 650)
        
        # Hard-coded pose2d value
        self.pose2d = "rtmo"
        
        # Initialize the UI
        self.init_ui()
        
    def init_ui(self):
        # Create a tab widget to separate the two models
        self.tab_widget = QTabWidget()
        
        # Create tabs for each model
        cnn_transformer_tab = QWidget()
        cnn_lstm_tab = QWidget()
        
        # Set up the CNN Transformer tab
        self.setup_cnn_transformer_tab(cnn_transformer_tab)
        
        # Set up the CNN LSTM tab
        self.setup_cnn_lstm_tab(cnn_lstm_tab)
        
        # Add tabs to the widget
        self.tab_widget.addTab(cnn_transformer_tab, "CNN Transformer")
        self.tab_widget.addTab(cnn_lstm_tab, "CNN LSTM")
        
        # Set the tab widget as the central widget
        self.setCentralWidget(self.tab_widget)
        
        # Process for running the scripts
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_output)  # Handle standard output
        self.process.readyReadStandardError.connect(self.handle_error)    # Handle error output
        self.process.finished.connect(self.process_finished)             # Handle process completion
    
    def setup_cnn_transformer_tab(self, tab):
        # Main layout for CNN Transformer
        main_layout = QVBoxLayout()
        
        # Input video file selection
        file_layout = QHBoxLayout()
        self.transformer_video_path_label = QLabel("No video selected")
        self.transformer_video_path_label.setWordWrap(True)
        select_video_btn = QPushButton("Select Video")
        select_video_btn.clicked.connect(lambda: self.select_file(
            self.transformer_video_path_label, 
            "Select Video File", 
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        ))
        file_layout.addWidget(QLabel("Video:"))
        file_layout.addWidget(self.transformer_video_path_label, 1)
        file_layout.addWidget(select_video_btn)
        main_layout.addLayout(file_layout)
        
        # Model file selection
        model_layout = QHBoxLayout()
        self.transformer_model_path_label = QLabel("No model selected")
        self.transformer_model_path_label.setWordWrap(True)
        select_model_btn = QPushButton("Select Model")
        select_model_btn.clicked.connect(lambda: self.select_file(
            self.transformer_model_path_label, 
            "Select Model File", 
            "PyTorch Models (*.pth);;All Files (*)"
        ))
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.transformer_model_path_label, 1)
        model_layout.addWidget(select_model_btn)
        main_layout.addLayout(model_layout)
        
        # Label encoder selection
        label_layout = QHBoxLayout()
        self.transformer_label_path_label = QLabel("No label encoder selected")
        self.transformer_label_path_label.setWordWrap(True)
        select_label_btn = QPushButton("Select Label Encoder")
        select_label_btn.clicked.connect(lambda: self.select_file(
            self.transformer_label_path_label, 
            "Select Label Encoder File", 
            "NumPy Files (*.npy);;All Files (*)"
        ))
        label_layout.addWidget(QLabel("Label Encoder:"))
        label_layout.addWidget(self.transformer_label_path_label, 1)
        label_layout.addWidget(select_label_btn)
        main_layout.addLayout(label_layout)
        
        # Options
        options_layout = QVBoxLayout()
        
        # Display pose2d (as non-editable info)
        pose_layout = QHBoxLayout()
        pose_layout.addWidget(QLabel("Pose2D:"))
        pose_label = QLabel(self.pose2d)
        pose_label.setStyleSheet("font-weight: bold;")
        pose_layout.addWidget(pose_label)
        pose_layout.addStretch(1)
        options_layout.addLayout(pose_layout)
        
        # Device options
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.transformer_device_combo = QComboBox()
        self.transformer_device_combo.addItems(["mps", "cpu", "cuda"])  # Add device options
        device_layout.addWidget(self.transformer_device_combo)
        options_layout.addLayout(device_layout)
        
        # Bbox threshold
        bbox_layout = QHBoxLayout()
        bbox_layout.addWidget(QLabel("Bbox Threshold:"))
        self.transformer_bbox_thr = QDoubleSpinBox()
        self.transformer_bbox_thr.setRange(0.1, 1.0)  # Set range for bbox threshold
        self.transformer_bbox_thr.setSingleStep(0.1)  # Set step size
        self.transformer_bbox_thr.setValue(0.6)       # Set default value
        bbox_layout.addWidget(self.transformer_bbox_thr)
        options_layout.addLayout(bbox_layout)
        
        # Checkboxes
        check_layout = QHBoxLayout()
        self.transformer_show_checkbox = QCheckBox("Show")  # Checkbox for showing output
        self.transformer_show_checkbox.setChecked(True)     # Default checked
        self.transformer_draw_bbox_checkbox = QCheckBox("Draw Bbox")  # Checkbox for drawing bbox
        self.transformer_draw_bbox_checkbox.setChecked(True)          # Default checked
        check_layout.addWidget(self.transformer_show_checkbox)
        check_layout.addWidget(self.transformer_draw_bbox_checkbox)
        options_layout.addLayout(check_layout)
        
        main_layout.addLayout(options_layout)
        
        # Output section - Using QTextEdit for real-time terminal output
        main_layout.addWidget(QLabel("Output:"))
        self.transformer_output_text = QTextEdit()
        self.transformer_output_text.setReadOnly(True)  # Make output read-only
        self.transformer_output_text.setStyleSheet(
            "background-color: #f0f0f0; font-family: 'Courier New', monospace;"
        )
        self.transformer_output_text.setText("Ready")  # Default text
        self.transformer_output_text.setMinimumHeight(200)  # Set minimum height
        main_layout.addWidget(self.transformer_output_text)
        
        # Run button
        self.transformer_run_button = QPushButton("Run CNN Transformer")
        self.transformer_run_button.setStyleSheet("font-size: 16px; padding: 10px;")
        self.transformer_run_button.clicked.connect(self.run_cnn_transformer)  # Connect to run function
        main_layout.addWidget(self.transformer_run_button)
        
        tab.setLayout(main_layout)  # Set layout for the tab
    
    def setup_cnn_lstm_tab(self, tab):
        # Main layout for CNN LSTM
        main_layout = QVBoxLayout()
        
        # Input video file selection
        file_layout = QHBoxLayout()
        self.lstm_video_path_label = QLabel("No video selected")
        self.lstm_video_path_label.setWordWrap(True)
        select_video_btn = QPushButton("Select Video")
        select_video_btn.clicked.connect(lambda: self.select_file(
            self.lstm_video_path_label, 
            "Select Video File", 
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        ))
        file_layout.addWidget(QLabel("Video:"))
        file_layout.addWidget(self.lstm_video_path_label, 1)
        file_layout.addWidget(select_video_btn)
        main_layout.addLayout(file_layout)
        
        # Model file selection
        model_layout = QHBoxLayout()
        self.lstm_model_path_label = QLabel("No model selected")
        self.lstm_model_path_label.setWordWrap(True)
        select_model_btn = QPushButton("Select Model")
        select_model_btn.clicked.connect(lambda: self.select_file(
            self.lstm_model_path_label, 
            "Select Model File", 
            "H5 Files (*.h5);;All Files (*)"
        ))
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.lstm_model_path_label, 1)
        model_layout.addWidget(select_model_btn)
        main_layout.addLayout(model_layout)
        
        # Label encoder selection
        label_layout = QHBoxLayout()
        self.lstm_label_path_label = QLabel("No label encoder selected")
        self.lstm_label_path_label.setWordWrap(True)
        select_label_btn = QPushButton("Select Label Encoder")
        select_label_btn.clicked.connect(lambda: self.select_file(
            self.lstm_label_path_label, 
            "Select Label Encoder File", 
            "Text Files (*.txt);;All Files (*)"
        ))
        label_layout.addWidget(QLabel("Label Encoder:"))
        label_layout.addWidget(self.lstm_label_path_label, 1)
        label_layout.addWidget(select_label_btn)
        main_layout.addLayout(label_layout)
        
        # Options
        options_layout = QVBoxLayout()
        
        # Display pose2d (as non-editable info)
        pose_layout = QHBoxLayout()
        pose_layout.addWidget(QLabel("Pose2D:"))
        pose_label = QLabel(self.pose2d)
        pose_label.setStyleSheet("font-weight: bold;")
        pose_layout.addWidget(pose_label)
        pose_layout.addStretch(1)
        options_layout.addLayout(pose_layout)
        
        # Device options
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.lstm_device_combo = QComboBox()
        self.lstm_device_combo.addItems(["mps", "cpu", "cuda"])  # Add device options
        device_layout.addWidget(self.lstm_device_combo)
        options_layout.addLayout(device_layout)
        
        # Bbox threshold
        bbox_layout = QHBoxLayout()
        bbox_layout.addWidget(QLabel("Bbox Threshold:"))
        self.lstm_bbox_thr = QDoubleSpinBox()
        self.lstm_bbox_thr.setRange(0.1, 1.0)  # Set range for bbox threshold
        self.lstm_bbox_thr.setSingleStep(0.1)  # Set step size
        self.lstm_bbox_thr.setValue(0.6)       # Set default value
        bbox_layout.addWidget(self.lstm_bbox_thr)
        options_layout.addLayout(bbox_layout)
        
        # Checkboxes
        check_layout = QHBoxLayout()
        self.lstm_show_checkbox = QCheckBox("Show")  # Checkbox for showing output
        self.lstm_show_checkbox.setChecked(True)     # Default checked
        self.lstm_draw_bbox_checkbox = QCheckBox("Draw Bbox")  # Checkbox for drawing bbox
        self.lstm_draw_bbox_checkbox.setChecked(True)          # Default checked
        check_layout.addWidget(self.lstm_show_checkbox)
        check_layout.addWidget(self.lstm_draw_bbox_checkbox)
        options_layout.addLayout(check_layout)
        
        main_layout.addLayout(options_layout)
        
        # Output section - Using QTextEdit for real-time terminal output
        main_layout.addWidget(QLabel("Output:"))
        self.lstm_output_text = QTextEdit()
        self.lstm_output_text.setReadOnly(True)  # Make output read-only
        self.lstm_output_text.setStyleSheet(
            "background-color: #f0f0f0; font-family: 'Courier New', monospace;"
        )
        self.lstm_output_text.setText("Ready")  # Default text
        self.lstm_output_text.setMinimumHeight(200)  # Set minimum height
        main_layout.addWidget(self.lstm_output_text)
        
        # Run button
        self.lstm_run_button = QPushButton("Run CNN LSTM")
        self.lstm_run_button.setStyleSheet("font-size: 16px; padding: 10px;")
        self.lstm_run_button.clicked.connect(self.run_cnn_lstm)  # Connect to run function
        main_layout.addWidget(self.lstm_run_button)
        
        tab.setLayout(main_layout)  # Set layout for the tab
    
    def select_file(self, label_widget, title, file_filter):
        # Open a file dialog to select a file and update the label with the selected file path
        file_path, _ = QFileDialog.getOpenFileName(self, title, '', file_filter)
        if file_path:
            label_widget.setText(file_path)
    
    def run_cnn_transformer(self):
        # Gather input paths and options for CNN Transformer inference
        video_path = self.transformer_video_path_label.text()
        model_path = self.transformer_model_path_label.text()
        label_path = self.transformer_label_path_label.text()
        
        # Validate that all required files are selected
        if video_path == "No video selected" or model_path == "No model selected" or label_path == "No label encoder selected":
            self.transformer_output_text.setText("Error: Please select all required files")
            return
        
        # Build the command for running the CNN Transformer inference script
        cmd = [
            "python", 
            "cnn_transformer/cnn_transformer_inference.py",
            video_path,
            "--pose2d", self.pose2d,
            "--device", self.transformer_device_combo.currentText(),
            "--model-path", model_path,
            "--label-encoder", label_path,
            "--bbox-thr", str(self.transformer_bbox_thr.value())
        ]
        
        # Add optional flags based on user selections
        if self.transformer_show_checkbox.isChecked():
            cmd.append("--show")
        
        if self.transformer_draw_bbox_checkbox.isChecked():
            cmd.append("--draw-bbox")
        
        # Update UI to indicate the process is running
        self.transformer_run_button.setEnabled(False)
        self.lstm_run_button.setEnabled(False)
        self.transformer_output_text.clear()
        self.transformer_output_text.append("Running command: " + " ".join(cmd))
        self.transformer_output_text.append("\nOutput:")
        self.current_output_widget = self.transformer_output_text
        
        # Start the process
        self.process.start("python", cmd[1:])
    
    def run_cnn_lstm(self):
        # Gather input paths and options for CNN LSTM inference
        video_path = self.lstm_video_path_label.text()
        model_path = self.lstm_model_path_label.text()
        label_path = self.lstm_label_path_label.text()
        
        # Validate that all required files are selected
        if video_path == "No video selected" or model_path == "No model selected" or label_path == "No label encoder selected":
            self.lstm_output_text.setText("Error: Please select all required files")
            return
        
        # Build the command for running the CNN LSTM inference script
        cmd = [
            "python", 
            "cnn_lstm/cnn_lstm_inference.py",
            video_path,
            "--pose2d", self.pose2d,
            "--device", self.lstm_device_combo.currentText(),
            "--model-path", model_path,
            "--label-encoder", label_path,
            "--bbox-thr", str(self.lstm_bbox_thr.value())
        ]
        
        # Add optional flags based on user selections
        if self.lstm_show_checkbox.isChecked():
            cmd.append("--show")
        
        if self.lstm_draw_bbox_checkbox.isChecked():
            cmd.append("--draw-bbox")
        
        # Update UI to indicate the process is running
        self.transformer_run_button.setEnabled(False)
        self.lstm_run_button.setEnabled(False)
        self.lstm_output_text.clear()
        self.lstm_output_text.append("Running command: " + " ".join(cmd))
        self.lstm_output_text.append("\nOutput:")
        self.current_output_widget = self.lstm_output_text
        
        # Start the process
        self.process.start("python", cmd[1:])
    
    def handle_output(self):
        # Handle standard output from the process and display it in the output widget
        data = self.process.readAllStandardOutput().data().decode()
        self.current_output_widget.append(data)
        # Scroll to the bottom to show the latest output
        self.current_output_widget.verticalScrollBar().setValue(
            self.current_output_widget.verticalScrollBar().maximum()
        )
    
    def handle_error(self):
        # Handle error output from the process and display it in the output widget
        data = self.process.readAllStandardError().data().decode()
        self.current_output_widget.append("Error: " + data)
        # Scroll to the bottom to show the latest output
        self.current_output_widget.verticalScrollBar().setValue(
            self.current_output_widget.verticalScrollBar().maximum()
        )
    
    def process_finished(self):
        # Handle process completion and re-enable the run buttons
        self.transformer_run_button.setEnabled(True)
        self.lstm_run_button.setEnabled(True)
        self.current_output_widget.append("\n✅ Process Finished")
        # Scroll to the bottom to show the latest output
        self.current_output_widget.verticalScrollBar().setValue(
            self.current_output_widget.verticalScrollBar().maximum()
        )

if __name__ == "__main__":
    # Entry point for the application
    app = QApplication(sys.argv)
    window = ModelInferenceGUI()
    window.show()
    sys.exit(app.exec_())