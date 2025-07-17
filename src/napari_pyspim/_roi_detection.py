"""
ROI detection widget for automated region of interest detection and cropping.
"""

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QComboBox, QLabel, QGroupBox, QMessageBox,
    QProgressBar
)
from qtpy.QtCore import Signal, QThread
from PyQt5.QtCore import pyqtSignal

from pyspim import roi


class RoiDetectionWorker(QThread):
    """Worker thread for ROI detection."""
    
    roi_detected = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, a_raw, b_raw, method='otsu'):
        super().__init__()
        self.a_raw = a_raw
        self.b_raw = b_raw
        self.method = method
        
    def run(self):
        """Detect ROIs in background thread."""
        try:
            # Find ROIs for images A & B
            roia = roi.detect_roi_3d(self.a_raw, self.method)
            roib = roi.detect_roi_3d(self.b_raw, self.method)
            roic = roi.combine_rois(roia, roib)
            
            # Apply ROI cropping
            a_cropped = self.a_raw[
                roic[0][0]:roic[0][1],
                roic[1][0]:roic[1][1],
                roic[2][0]:roic[2][1]
            ]
            b_cropped = self.b_raw[
                roic[0][0]:roic[0][1],
                roic[1][0]:roic[1][1],
                roic[2][0]:roic[2][1]
            ]
            
            result = {
                'a_cropped': a_cropped,
                'b_cropped': b_cropped,
                'roi_coords': roic,
                'method': self.method
            }
            
            self.roi_detected.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class RoiDetectionWidget(QWidget):
    """Widget for ROI detection and cropping."""
    
    roi_applied = Signal(dict)
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.worker = None
        self.input_data = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        
        # Method selection
        method_group = QGroupBox("ROI Detection Method")
        method_layout = QFormLayout()
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(['otsu', 'triangle', 'manual'])
        self.method_combo.setCurrentText('otsu')
        
        method_layout.addRow("Threshold Method:", self.method_combo)
        method_group.setLayout(method_layout)
        
        # Detection button
        self.detect_button = QPushButton("Detect and Apply ROI")
        self.detect_button.clicked.connect(self.detect_roi)
        self.detect_button.setEnabled(False)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Status label
        self.status_label = QLabel("No input data available")
        
        # ROI info display
        self.roi_info_label = QLabel("")
        self.roi_info_label.setWordWrap(True)
        
        # Add widgets to layout
        layout.addWidget(method_group)
        layout.addWidget(self.detect_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.roi_info_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def set_input_data(self, data_dict):
        """Set input data from previous step."""
        self.input_data = data_dict
        self.detect_button.setEnabled(True)
        self.status_label.setText("Input data ready for ROI detection")
        
        # Update ROI info if we have previous results
        if 'roi_coords' in data_dict:
            self._update_roi_info(data_dict['roi_coords'])
            
    def detect_roi(self):
        """Detect ROI using selected method."""
        if not self.input_data:
            QMessageBox.warning(self, "Error", "No input data available")
            return
            
        a_raw = self.input_data['a_raw']
        b_raw = self.input_data['b_raw']
        method = self.method_combo.currentText()
        
        self.detect_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Detecting ROI...")
        
        # Create and start worker
        self.worker = RoiDetectionWorker(a_raw, b_raw, method)
        self.worker.roi_detected.connect(self.on_roi_detected)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
        
    def on_roi_detected(self, result):
        """Handle successful ROI detection."""
        a_cropped = result['a_cropped']
        b_cropped = result['b_cropped']
        roi_coords = result['roi_coords']
        
        # Remove old layers if they exist
        for layer_name in ['A_cropped', 'B_cropped']:
            try:
                layer = self.viewer.layers[layer_name]
                self.viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass
        
        # Add cropped data as new layers
        self.viewer.add_image(
            a_cropped,
            name="A_cropped",
            metadata={
                'roi_coords': roi_coords,
                'method': result['method'],
                'original_shape': self.input_data['a_raw'].shape
            }
        )
        
        self.viewer.add_image(
            b_cropped,
            name="B_cropped",
            metadata={
                'roi_coords': roi_coords,
                'method': result['method'],
                'original_shape': self.input_data['b_raw'].shape
            }
        )
        
        # Update status and info
        self.status_label.setText(
            f"ROI applied successfully! A: {a_cropped.shape}, B: {b_cropped.shape}"
        )
        self._update_roi_info(roi_coords)
        
        self.detect_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Emit signal with cropped data
        output_data = {
            'a_cropped': a_cropped,
            'b_cropped': b_cropped,
            'roi_coords': roi_coords,
            'method': result['method']
        }
        self.roi_applied.emit(output_data)
        
    def on_error(self, error_msg):
        """Handle ROI detection error."""
        QMessageBox.critical(self, "Error", f"ROI detection failed: {error_msg}")
        self.status_label.setText("Error during ROI detection")
        self.detect_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def _update_roi_info(self, roi_coords):
        """Update ROI information display."""
        if roi_coords:
            info_text = f"""
            <b>ROI Coordinates:</b><br>
            X: {roi_coords[0][0]} to {roi_coords[0][1]} ({roi_coords[0][1] - roi_coords[0][0]} pixels)<br>
            Y: {roi_coords[1][0]} to {roi_coords[1][1]} ({roi_coords[1][1] - roi_coords[1][0]} pixels)<br>
            Z: {roi_coords[2][0]} to {roi_coords[2][1]} ({roi_coords[2][1] - roi_coords[2][0]} pixels)
            """
            self.roi_info_label.setText(info_text)
        else:
            self.roi_info_label.setText("") 