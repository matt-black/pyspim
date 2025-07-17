"""
Data loader widget for loading μManager acquisitions and setting parameters.
"""

import os
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QLabel, QFileDialog, QGroupBox, QMessageBox
)
from qtpy.QtCore import Signal, QThread
from PyQt5.QtCore import pyqtSignal
import numpy as np

from pyspim.data import dispim as data


class DataLoaderWorker(QThread):
    """Worker thread for loading data."""
    
    data_loaded = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, data_path, camera_offset=100):
        super().__init__()
        self.data_path = data_path
        self.camera_offset = camera_offset
        
    def run(self):
        """Load the data in background thread."""
        try:
            with data.uManagerAcquisition(self.data_path, False, np) as acq:
                a_raw = acq.get('a', 0, 0)
                b_raw = acq.get('b', 0, 0)
                
            # Apply camera offset
            a_raw = data.subtract_constant_uint16arr(a_raw, self.camera_offset)
            b_raw = data.subtract_constant_uint16arr(b_raw, self.camera_offset)
            
            result = {
                'a_raw': a_raw,
                'b_raw': b_raw,
                'data_path': self.data_path
            }
            
            self.data_loaded.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class DataLoaderWidget(QWidget):
    """Widget for loading μManager acquisition data."""
    
    data_loaded = Signal(dict)
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.worker = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        
        # Data path selection
        path_group = QGroupBox("Data Path")
        path_layout = QFormLayout()
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select μManager acquisition folder...")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_data_path)
        
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_button)
        
        path_layout.addRow("Data Path:", path_row)
        path_group.setLayout(path_layout)
        
        # Acquisition parameters
        params_group = QGroupBox("Acquisition Parameters")
        params_layout = QFormLayout()
        
        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(0.1, 10.0)
        self.step_size_spin.setValue(0.5)
        self.step_size_spin.setSuffix(" μm")
        self.step_size_spin.setDecimals(3)
        
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.01, 1.0)
        self.pixel_size_spin.setValue(0.1625)
        self.pixel_size_spin.setSuffix(" μm")
        self.pixel_size_spin.setDecimals(4)
        
        self.theta_spin = QDoubleSpinBox()
        self.theta_spin.setRange(0, 90)
        self.theta_spin.setValue(45)
        self.theta_spin.setSuffix("°")
        self.theta_spin.setDecimals(1)
        
        self.camera_offset_spin = QSpinBox()
        self.camera_offset_spin.setRange(0, 1000)
        self.camera_offset_spin.setValue(100)
        
        params_layout.addRow("Step Size:", self.step_size_spin)
        params_layout.addRow("Pixel Size:", self.pixel_size_spin)
        params_layout.addRow("Theta (angle):", self.theta_spin)
        params_layout.addRow("Camera Offset:", self.camera_offset_spin)
        params_group.setLayout(params_layout)
        
        # Load button
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        self.load_button.setEnabled(False)
        
        # Status label
        self.status_label = QLabel("Ready to load data")
        
        # Add widgets to layout
        layout.addWidget(path_group)
        layout.addWidget(params_group)
        layout.addWidget(self.load_button)
        layout.addWidget(self.status_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Connect path validation
        self.path_edit.textChanged.connect(self.validate_path)
        
    def browse_data_path(self):
        """Open file dialog to select data path."""
        path = QFileDialog.getExistingDirectory(
            self, "Select μManager Acquisition Folder"
        )
        if path:
            self.path_edit.setText(path)
            
    def validate_path(self):
        """Validate the selected data path."""
        path = self.path_edit.text()
        if path and os.path.exists(path):
            self.load_button.setEnabled(True)
            self.status_label.setText("Path valid - ready to load")
        else:
            self.load_button.setEnabled(False)
            self.status_label.setText("Invalid path")
            
    def load_data(self):
        """Load the data using background worker."""
        data_path = self.path_edit.text()
        
        if not data_path or not os.path.exists(data_path):
            QMessageBox.warning(self, "Error", "Please select a valid data path")
            return
            
        self.load_button.setEnabled(False)
        self.status_label.setText("Loading data...")
        
        # Create and start worker
        self.worker = DataLoaderWorker(
            data_path, 
            self.camera_offset_spin.value()
        )
        self.worker.data_loaded.connect(self.on_data_loaded)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
        
    def on_data_loaded(self, result):
        """Handle successful data loading."""
        # Add data to napari viewer
        a_raw = result['a_raw']
        b_raw = result['b_raw']
        
        # Add as layers with metadata
        self.viewer.add_image(
            a_raw, 
            name="A_raw", 
            metadata={
                'data_path': result['data_path'],
                'step_size': self.step_size_spin.value(),
                'pixel_size': self.pixel_size_spin.value(),
                'theta': self.theta_spin.value() * np.pi / 180,  # Convert to radians
                'camera_offset': self.camera_offset_spin.value()
            }
        )
        
        self.viewer.add_image(
            b_raw, 
            name="B_raw",
            metadata={
                'data_path': result['data_path'],
                'step_size': self.step_size_spin.value(),
                'pixel_size': self.pixel_size_spin.value(),
                'theta': self.theta_spin.value() * np.pi / 180,
                'camera_offset': self.camera_offset_spin.value()
            }
        )
        
        self.status_label.setText(
            f"Data loaded successfully! A: {a_raw.shape}, B: {b_raw.shape}"
        )
        self.load_button.setEnabled(True)
        
        # Emit signal with loaded data
        self.data_loaded.emit(result)
        
    def on_error(self, error_msg):
        """Handle loading error."""
        QMessageBox.critical(self, "Error", f"Failed to load data: {error_msg}")
        self.status_label.setText("Error loading data")
        self.load_button.setEnabled(True) 