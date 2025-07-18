"""
ROI detection widget for automated and manual region of interest detection and cropping.
"""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QPushButton, QComboBox,
    QLabel, QGroupBox, QMessageBox, QProgressBar, QCheckBox,
    QRadioButton, QButtonGroup
)
from qtpy.QtCore import Signal, QThread
from PyQt5.QtCore import pyqtSignal

from pyspim import roi


class RoiDetectionWorker(QThread):
    """Worker thread for ROI detection."""
    
    roi_detected = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, a_raw, b_raw, method='otsu', process_single_channel=None):
        super().__init__()
        self.a_raw = a_raw
        self.b_raw = b_raw
        self.method = method
        # 'a', 'b', or None for both
        self.process_single_channel = process_single_channel
        
    def run(self):
        """Detect ROIs in background thread."""
        try:
            if self.process_single_channel == 'a':
                # Process only channel A
                roia = roi.detect_roi_3d(self.a_raw, self.method)
                roic = roia
                a_cropped = self.a_raw[
                    roic[0][0]:roic[0][1],
                    roic[1][0]:roic[1][1],
                    roic[2][0]:roic[2][1]
                ]
                b_cropped = None
                
            elif self.process_single_channel == 'b':
                # Process only channel B
                roib = roi.detect_roi_3d(self.b_raw, self.method)
                roic = roib
                b_cropped = self.b_raw[
                    roic[0][0]:roic[0][1],
                    roic[1][0]:roic[1][1],
                    roic[2][0]:roic[2][1]
                ]
                a_cropped = None
                
            else:
                # Process both channels
                roia = roi.detect_roi_3d(self.a_raw, self.method)
                roib = roi.detect_roi_3d(self.b_raw, self.method)
                roic = roi.combine_rois(roia, roib)
                
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
                'method': self.method,
                'process_single_channel': self.process_single_channel
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
        self.manual_roi_coords = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        
        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout()
        
        self.channel_group = QButtonGroup()
        self.both_channels_radio = QRadioButton("Process both channels (A & B)")
        self.channel_a_radio = QRadioButton("Process only channel A")
        self.channel_b_radio = QRadioButton("Process only channel B")
        
        self.channel_group.addButton(self.both_channels_radio, 0)
        self.channel_group.addButton(self.channel_a_radio, 1)
        self.channel_group.addButton(self.channel_b_radio, 2)
        
        self.both_channels_radio.setChecked(True)
        
        channel_layout.addWidget(self.both_channels_radio)
        channel_layout.addWidget(self.channel_a_radio)
        channel_layout.addWidget(self.channel_b_radio)
        channel_group.setLayout(channel_layout)
        
        # ROI method selection
        method_group = QGroupBox("ROI Detection Method")
        method_layout = QFormLayout()
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(['otsu', 'triangle', 'manual'])
        self.method_combo.setCurrentText('otsu')
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        
        method_layout.addRow("Threshold Method:", self.method_combo)
        method_group.setLayout(method_layout)
        
        # Manual ROI instructions
        self.manual_instructions = QLabel(
            "For manual ROI: Use napari's rectangle selection tool to draw a ROI, "
            "then click 'Apply Manual ROI'"
        )
        self.manual_instructions.setWordWrap(True)
        self.manual_instructions.setVisible(False)
        
        # Manual ROI button
        self.manual_roi_button = QPushButton("Apply Manual ROI")
        self.manual_roi_button.clicked.connect(self.apply_manual_roi)
        self.manual_roi_button.setVisible(False)
        self.manual_roi_button.setEnabled(False)
        
        # Skip ROI option
        self.skip_roi_checkbox = QCheckBox("Skip ROI detection (use full image)")
        self.skip_roi_checkbox.toggled.connect(self.on_skip_roi_toggled)
        
        # Skip ROI button
        self.skip_roi_button = QPushButton("Skip ROI and Continue")
        self.skip_roi_button.clicked.connect(self.apply_no_roi)
        self.skip_roi_button.setVisible(False)
        self.skip_roi_button.setEnabled(False)
        
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
        layout.addWidget(channel_group)
        layout.addWidget(method_group)
        layout.addWidget(self.manual_instructions)
        layout.addWidget(self.manual_roi_button)
        layout.addWidget(self.skip_roi_checkbox)
        layout.addWidget(self.skip_roi_button) # Added skip ROI button
        layout.addWidget(self.detect_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.roi_info_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def on_method_changed(self, method):
        """Handle method selection change."""
        is_manual = method == 'manual'
        self.manual_instructions.setVisible(is_manual)
        self.manual_roi_button.setVisible(is_manual)
        self.detect_button.setEnabled(not is_manual and self.input_data is not None)
        
    def on_skip_roi_toggled(self, checked):
        """Handle skip ROI checkbox toggle."""
        if checked:
            self.detect_button.setEnabled(False)
            self.method_combo.setEnabled(False)
            self.manual_roi_button.setEnabled(False)
            self.skip_roi_button.setVisible(True)
            self.skip_roi_button.setEnabled(self.input_data is not None)
        else:
            self.detect_button.setEnabled(self.input_data is not None and self.method_combo.currentText() != 'manual')
            self.method_combo.setEnabled(True)
            self.manual_roi_button.setEnabled(self.method_combo.currentText() == 'manual')
            self.skip_roi_button.setVisible(False)
            self.skip_roi_button.setEnabled(False)
        
    def set_input_data(self, data_dict):
        """Set input data from previous step."""
        self.input_data = data_dict
        is_manual = self.method_combo.currentText() == 'manual'
        self.detect_button.setEnabled(True and not is_manual)
        
        # Enable skip ROI button if checkbox is checked
        if self.skip_roi_checkbox.isChecked():
            self.skip_roi_button.setEnabled(True)
        
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
        
        # Determine which channel to process
        process_single_channel = None
        if self.channel_a_radio.isChecked():
            process_single_channel = 'a'
        elif self.channel_b_radio.isChecked():
            process_single_channel = 'b'
        
        self.detect_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Detecting ROI...")
        
        # Create and start worker
        self.worker = RoiDetectionWorker(a_raw, b_raw, method, process_single_channel)
        self.worker.roi_detected.connect(self.on_roi_detected)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
        
    def apply_manual_roi(self):
        """Apply manually selected ROI from napari viewer."""
        if not self.input_data:
            QMessageBox.warning(self, "Error", "No input data available")
            return
            
        # Check if there's a rectangle selection in the viewer
        shapes_layer = None
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and len(layer.data) > 0:
                shapes_layer = layer
                break
                
        if not shapes_layer or len(shapes_layer.data) == 0:
            msg = "Please draw a rectangle selection first using napari's rectangle tool"
            QMessageBox.warning(self, "Error", msg)
            return
            
        # Get the rectangle coordinates
        rect = shapes_layer.data[0]  # Assume first shape is the ROI
        if len(rect) != 4:
            QMessageBox.warning(self, "Error", "Please draw a rectangle (4 points)")
            return
            
        # Convert rectangle to ROI coordinates
        # Rectangle format: [x1, y1], [x2, y2], [x3, y3], [x4, y4]
        x_coords = [point[0] for point in rect]
        y_coords = [point[1] for point in rect]
        
        # Get current slice for Z coordinate
        # Assuming Z is first dimension
        current_z = self.viewer.dims.current_step[0]
        
        roi_coords = [
            (int(min(x_coords)), int(max(x_coords))),
            (int(min(y_coords)), int(max(y_coords))),
            (current_z, current_z + 1)  # Single Z slice for now
        ]
        
        # Apply ROI cropping
        a_raw = self.input_data['a_raw']
        b_raw = self.input_data['b_raw']
        
        # Determine which channel to process
        process_single_channel = None
        if self.channel_a_radio.isChecked():
            process_single_channel = 'a'
            a_cropped = a_raw[
                roi_coords[0][0]:roi_coords[0][1],
                roi_coords[1][0]:roi_coords[1][1],
                roi_coords[2][0]:roi_coords[2][1]
            ]
            b_cropped = None
        elif self.channel_b_radio.isChecked():
            process_single_channel = 'b'
            b_cropped = b_raw[
                roi_coords[0][0]:roi_coords[0][1],
                roi_coords[1][0]:roi_coords[1][1],
                roi_coords[2][0]:roi_coords[2][1]
            ]
            a_cropped = None
        else:
            # Process both channels
            a_cropped = a_raw[
                roi_coords[0][0]:roi_coords[0][1],
                roi_coords[1][0]:roi_coords[1][1],
                roi_coords[2][0]:roi_coords[2][1]
            ]
            b_cropped = b_raw[
                roi_coords[0][0]:roi_coords[0][1],
                roi_coords[1][0]:roi_coords[1][1],
                roi_coords[2][0]:roi_coords[2][1]
            ]
        
        # Remove the shapes layer
        self.viewer.layers.remove(shapes_layer)
        
        # Add cropped data as new layers
        if a_cropped is not None:
            self._add_cropped_layer(
                a_cropped, "A_cropped", roi_coords, "manual"
            )
        if b_cropped is not None:
            self._add_cropped_layer(
                b_cropped, "B_cropped", roi_coords, "manual"
            )
        
        # Update status and info
        self.status_label.setText("Manual ROI applied successfully!")
        self._update_roi_info(roi_coords)
        
        # Emit signal with cropped data
        output_data = {
            'a_cropped': a_cropped,
            'b_cropped': b_cropped,
            'roi_coords': roi_coords,
            'method': 'manual',
            'process_single_channel': process_single_channel
        }
        
        # Pass through parameters from input data
        for key in ['step_size', 'pixel_size', 'theta', 'camera_offset']:
            if key in self.input_data:
                output_data[key] = self.input_data[key]
        
        self.roi_applied.emit(output_data)
        
    def apply_no_roi(self):
        """Skip ROI detection and use full images."""
        a_raw = self.input_data['a_raw']
        b_raw = self.input_data['b_raw']
        
        # Determine which channel to process
        process_single_channel = None
        if self.channel_a_radio.isChecked():
            process_single_channel = 'a'
            a_cropped = a_raw.copy()
            b_cropped = None
        elif self.channel_b_radio.isChecked():
            process_single_channel = 'b'
            b_cropped = b_raw.copy()
            a_cropped = None
        else:
            # Process both channels
            a_cropped = a_raw.copy()
            b_cropped = b_raw.copy()
        
        # No ROI coordinates (full image)
        roi_coords = None
        
        # Add data as new layers
        if a_cropped is not None:
            self._add_cropped_layer(
                a_cropped, "A_full", roi_coords, "none"
            )
        if b_cropped is not None:
            self._add_cropped_layer(
                b_cropped, "B_full", roi_coords, "none"
            )
        
        # Update status and info
        self.status_label.setText("Using full images (no ROI applied)")
        self._update_roi_info(roi_coords)
        
        # Emit signal with data
        output_data = {
            'a_cropped': a_cropped,
            'b_cropped': b_cropped,
            'roi_coords': roi_coords,
            'method': 'none',
            'process_single_channel': process_single_channel
        }
        
        # Pass through parameters from input data
        for key in ['step_size', 'pixel_size', 'theta', 'camera_offset']:
            if key in self.input_data:
                output_data[key] = self.input_data[key]
        
        self.roi_applied.emit(output_data)
        
    def _add_cropped_layer(self, data, name, roi_coords, method):
        """Add cropped data as a new layer."""
        # Remove old layer if it exists
        try:
            layer = self.viewer.layers[name]
            self.viewer.layers.remove(layer)
        except (KeyError, ValueError):
            pass
        
        # Prepare metadata
        metadata = {
            'roi_coords': roi_coords,
            'method': method,
            'original_shape': self.input_data['a_raw'].shape if 'a_raw' in self.input_data else None
        }
        
        # Add acquisition parameters to metadata if available
        for key in ['step_size', 'pixel_size', 'theta', 'camera_offset']:
            if key in self.input_data:
                metadata[key] = self.input_data[key]
        
        # Add new layer
        self.viewer.add_image(
            data,
            name=name,
            metadata=metadata
        )
        
    def on_roi_detected(self, result):
        """Handle successful ROI detection."""
        a_cropped = result['a_cropped']
        b_cropped = result['b_cropped']
        roi_coords = result['roi_coords']
        
        # Add cropped data as new layers (only for processed channels)
        if a_cropped is not None:
            self._add_cropped_layer(
                a_cropped, "A_cropped", roi_coords, result['method']
            )
        if b_cropped is not None:
            self._add_cropped_layer(
                b_cropped, "B_cropped", roi_coords, result['method']
            )
        
        # Update status and info
        if a_cropped is not None and b_cropped is not None:
            status_text = (
                f"ROI applied successfully! A: {a_cropped.shape}, "
                f"B: {b_cropped.shape}"
            )
        elif a_cropped is not None:
            status_text = f"ROI applied successfully! A: {a_cropped.shape}"
        else:
            status_text = f"ROI applied successfully! B: {b_cropped.shape}"
            
        self.status_label.setText(status_text)
        self._update_roi_info(roi_coords)
        
        self.detect_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Emit signal with cropped data and pass through parameters
        output_data = result.copy()
        # Pass through parameters from input data
        for key in ['step_size', 'pixel_size', 'theta', 'camera_offset']:
            if key in self.input_data:
                output_data[key] = self.input_data[key]
        
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
            self.roi_info_label.setText("Using full image (no ROI)") 