"""
Deskewing widget for deskewing dual-view SPIM data.
"""

import numpy as np
import math
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QPushButton, QComboBox,
    QLabel, QGroupBox, QMessageBox, QProgressBar, QCheckBox
)
from qtpy.QtCore import Signal, QThread
from PyQt5.QtCore import pyqtSignal

from pyspim import deskew as dsk
from pyspim import roi


class DeskewingWorker(QThread):
    """Worker thread for deskewing."""
    
    deskewed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, a_raw, b_raw, pixel_size, step_size, theta, 
                 method='orthogonal', recrop=False, process_single_channel=None):
        super().__init__()
        self.a_raw = a_raw
        self.b_raw = b_raw
        self.pixel_size = pixel_size
        self.step_size = step_size
        self.theta = theta
        self.method = method
        self.recrop = recrop
        # 'a', 'b', or None for both
        self.process_single_channel = process_single_channel
        
    def run(self):
        """Deskew data in background thread."""
        try:
            # Calculate step sizes
            step_size_lat = self.step_size / math.cos(self.theta)
            
            if self.process_single_channel == 'a':
                # Process only channel A
                a_dsk = dsk.deskew_stage_scan(
                    self.a_raw, self.pixel_size, step_size_lat, 1,
                    method=self.method
                )
                b_dsk = None
                
            elif self.process_single_channel == 'b':
                # Process only channel B
                b_dsk = dsk.deskew_stage_scan(
                    self.b_raw, self.pixel_size, step_size_lat, -1,
                    method=self.method
                )
                a_dsk = None
                
            else:
                # Process both channels
                # Deskew head A (direction = 1)
                a_dsk = dsk.deskew_stage_scan(
                    self.a_raw, self.pixel_size, step_size_lat, 1,
                    method=self.method
                )
                
                # Deskew head B (direction = -1)
                b_dsk = dsk.deskew_stage_scan(
                    self.b_raw, self.pixel_size, step_size_lat, -1,
                    method=self.method
                )
            
            # Optional re-cropping
            if self.recrop:
                if a_dsk is not None:
                    roia = roi.detect_roi_3d(a_dsk, 'triangle')
                    a_dsk = a_dsk[
                        roia[0][0]:roia[0][1],
                        roia[1][0]:roia[1][1],
                        roia[2][0]:roia[2][1]
                    ].astype(np.float32)
                
                if b_dsk is not None:
                    roib = roi.detect_roi_3d(b_dsk, 'triangle')
                    b_dsk = b_dsk[
                        roib[0][0]:roib[0][1],
                        roib[1][0]:roib[1][1],
                        roib[2][0]:roib[2][1]
                    ].astype(np.float32)
            else:
                if a_dsk is not None:
                    a_dsk = a_dsk.astype(np.float32)
                if b_dsk is not None:
                    b_dsk = b_dsk.astype(np.float32)
            
            result = {
                'a_deskewed': a_dsk,
                'b_deskewed': b_dsk,
                'method': self.method,
                'recrop': self.recrop,
                'step_size_lat': step_size_lat,
                'process_single_channel': self.process_single_channel
            }
            
            self.deskewed.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class DeskewingWidget(QWidget):
    """Widget for deskewing dual-view data."""
    
    deskewed = Signal(dict)
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.worker = None
        self.input_data = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        
        # Deskewing parameters
        params_group = QGroupBox("Deskewing Parameters")
        params_layout = QFormLayout()
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(['orthogonal'])
        self.method_combo.setCurrentText('orthogonal')
        
        self.recrop_check = QCheckBox("Re-crop after deskewing")
        self.recrop_check.setChecked(False)
        
        # These will be populated from previous step metadata
        self.step_size_label = QLabel("Not set")
        self.pixel_size_label = QLabel("Not set")
        self.theta_label = QLabel("Not set")
        
        params_layout.addRow("Method:", self.method_combo)
        params_layout.addRow("Step Size:", self.step_size_label)
        params_layout.addRow("Pixel Size:", self.pixel_size_label)
        params_layout.addRow("Theta:", self.theta_label)
        params_layout.addRow("Re-crop:", self.recrop_check)
        params_group.setLayout(params_layout)
        
        # Deskew button
        self.deskew_button = QPushButton("Deskew Data")
        self.deskew_button.clicked.connect(self.deskew_data)
        self.deskew_button.setEnabled(False)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Status label
        self.status_label = QLabel("No input data available")
        
        # Results info
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        
        # Add widgets to layout
        layout.addWidget(params_group)
        layout.addWidget(self.deskew_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.results_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def set_input_data(self, data_dict):
        """Set input data from previous step."""
        self.input_data = data_dict
        
        # Try to get parameters from napari layers
        self._update_parameters_from_layers()
        
        self.deskew_button.setEnabled(True)
        self.status_label.setText("Input data ready for deskewing")
        
    def _update_parameters_from_layers(self):
        """Update parameters from napari layer metadata."""
        # First try to get parameters from input data if available
        if self.input_data:
            # Check if parameters are passed through from previous step
            if 'step_size' in self.input_data:
                step_size = self.input_data['step_size']
                pixel_size = self.input_data.get('pixel_size', 'Not set')
                theta_rad = self.input_data.get('theta', 'Not set')
                
                self.step_size_label.setText(f"{step_size} μm")
                self.pixel_size_label.setText(f"{pixel_size} μm")
                
                if theta_rad != 'Not set':
                    theta_deg = theta_rad * 180 / math.pi
                    self.theta_label.setText(f"{theta_deg:.1f}°")
                else:
                    self.theta_label.setText("Not set")
                return
        
        # If not in input data, try to get from napari layers
        try:
            # Look for various possible layer names
            layer_names_to_check = [
                'A_cropped', 'B_cropped', 'A_full', 'B_full', 
                'A_raw', 'B_raw', 'A_deskewed', 'B_deskewed'
            ]
            
            for layer_name in layer_names_to_check:
                if layer_name in self.viewer.layers:
                    layer = self.viewer.layers[layer_name]
                    metadata = layer.metadata
                    
                    if metadata:
                        step_size = metadata.get('step_size', 'Not set')
                        pixel_size = metadata.get('pixel_size', 'Not set')
                        theta_rad = metadata.get('theta', 'Not set')
                        
                        if step_size != 'Not set':
                            self.step_size_label.setText(f"{step_size} μm")
                            self.pixel_size_label.setText(f"{pixel_size} μm")
                            
                            if theta_rad != 'Not set':
                                theta_deg = theta_rad * 180 / math.pi
                                self.theta_label.setText(f"{theta_deg:.1f}°")
                            else:
                                self.theta_label.setText("Not set")
                            return
        except Exception:
            pass
        
        # Fallback: Set default values if no parameters found
        self.step_size_label.setText("0.5 μm")
        self.pixel_size_label.setText("0.1625 μm")
        self.theta_label.setText("45.0°")
            
    def deskew_data(self):
        """Deskew the data using background worker."""
        if not self.input_data:
            QMessageBox.warning(self, "Error", "No input data available")
            return
            
        # Get parameters from labels with fallback to input data
        try:
            step_size_text = self.step_size_label.text()
            pixel_size_text = self.pixel_size_label.text()
            theta_text = self.theta_label.text()
            
            # Try to extract from labels first
            if step_size_text != "Not set":
                step_size = float(step_size_text.split()[0])
            else:
                # Fallback to input data
                step_size = self.input_data.get('step_size', 0.5)
                
            if pixel_size_text != "Not set":
                pixel_size = float(pixel_size_text.split()[0])
            else:
                # Fallback to input data
                pixel_size = self.input_data.get('pixel_size', 0.1625)
                
            if theta_text != "Not set":
                theta_deg = float(theta_text.split()[0])
                theta_rad = theta_deg * math.pi / 180
            else:
                # Fallback to input data
                theta_rad = self.input_data.get('theta', 45 * math.pi / 180)
                
        except (ValueError, IndexError):
            # Final fallback to input data or defaults
            step_size = self.input_data.get('step_size', 0.5)
            pixel_size = self.input_data.get('pixel_size', 0.1625)
            theta_rad = self.input_data.get('theta', 45 * math.pi / 180)
            
        a_raw = self.input_data.get('a_cropped')
        b_raw = self.input_data.get('b_cropped')
        method = self.method_combo.currentText()
        recrop = self.recrop_check.isChecked()
        
        # Get single channel processing info from input data
        process_single_channel = self.input_data.get('process_single_channel')
        
        self.deskew_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Deskewing data...")
        
        # Create and start worker
        self.worker = DeskewingWorker(
            a_raw, b_raw, pixel_size, step_size, theta_rad,
            method, recrop, process_single_channel
        )
        self.worker.deskewed.connect(self.on_deskewed)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
        
    def on_deskewed(self, result):
        """Handle successful deskewing."""
        a_deskewed = result['a_deskewed']
        b_deskewed = result['b_deskewed']
        
        # Remove old layers if they exist
        for layer_name in ['A_deskewed', 'B_deskewed']:
            try:
                layer = self.viewer.layers[layer_name]
                self.viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass
        
        # Add deskewed data as new layers (only for processed channels)
        if a_deskewed is not None:
            self.viewer.add_image(
                a_deskewed,
                name="A_deskewed",
                metadata={
                    'method': result['method'],
                    'recrop': result['recrop'],
                    'step_size_lat': result['step_size_lat']
                }
            )
        
        if b_deskewed is not None:
            self.viewer.add_image(
                b_deskewed,
                name="B_deskewed",
                metadata={
                    'method': result['method'],
                    'recrop': result['recrop'],
                    'step_size_lat': result['step_size_lat']
                }
            )
        
        # Update status and results
        if a_deskewed is not None and b_deskewed is not None:
            status_text = (
                f"Deskewing completed! A: {a_deskewed.shape}, "
                f"B: {b_deskewed.shape}"
            )
            results_text = f"""
            <b>Deskewing Results:</b><br>
            Method: {result['method']}<br>
            Re-cropped: {result['recrop']}<br>
            Lateral step size: {result['step_size_lat']:.3f} μm<br>
            A shape: {a_deskewed.shape}<br>
            B shape: {b_deskewed.shape}
            """
        elif a_deskewed is not None:
            status_text = f"Deskewing completed! A: {a_deskewed.shape}"
            results_text = f"""
            <b>Deskewing Results:</b><br>
            Method: {result['method']}<br>
            Re-cropped: {result['recrop']}<br>
            Lateral step size: {result['step_size_lat']:.3f} μm<br>
            A shape: {a_deskewed.shape}<br>
            B: Not processed
            """
        else:
            status_text = f"Deskewing completed! B: {b_deskewed.shape}"
            results_text = f"""
            <b>Deskewing Results:</b><br>
            Method: {result['method']}<br>
            Re-cropped: {result['recrop']}<br>
            Lateral step size: {result['step_size_lat']:.3f} μm<br>
            A: Not processed<br>
            B shape: {b_deskewed.shape}
            """
            
        self.status_label.setText(status_text)
        self.results_label.setText(results_text)
        
        self.deskew_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Emit signal with deskewed data
        output_data = {
            'a_deskewed': a_deskewed,
            'b_deskewed': b_deskewed,
            'method': result['method'],
            'recrop': result['recrop'],
            'step_size_lat': result['step_size_lat'],
            'process_single_channel': result.get('process_single_channel')
        }
        self.deskewed.emit(output_data)
        
    def on_error(self, error_msg):
        """Handle deskewing error."""
        QMessageBox.critical(self, "Error", f"Deskewing failed: {error_msg}")
        self.status_label.setText("Error during deskewing")
        self.deskew_button.setEnabled(True)
        self.progress_bar.setVisible(False) 