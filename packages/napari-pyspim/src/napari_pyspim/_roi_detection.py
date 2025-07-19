"""
ROI detection widget for automated and manual region of interest detection and cropping.
"""

import math
from PyQt5.QtCore import pyqtSignal
from qtpy.QtCore import QThread
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Lazy import to avoid CUDA compilation at module level
# from pyspim import roi


class RoiDetectionWorker(QThread):
    """Worker thread for ROI detection."""

    roi_detected = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, a_raw, b_raw, method="otsu", process_single_channel=None):
        super().__init__()
        self.a_raw = a_raw
        self.b_raw = b_raw
        self.method = method
        # 'a', 'b', or None for both
        self.process_single_channel = process_single_channel

    def run(self):
        """Detect ROIs in background thread."""
        try:
            # Lazy import to avoid CUDA compilation at module level
            from pyspim import roi

            if self.process_single_channel == "a":
                # Process only channel A
                roia = roi.detect_roi_3d(self.a_raw, self.method)
                roic = roia
                a_cropped = self.a_raw[
                    roic[0][0] : roic[0][1],
                    roic[1][0] : roic[1][1],
                    roic[2][0] : roic[2][1],
                ]
                b_cropped = None

            elif self.process_single_channel == "b":
                # Process only channel B
                roib = roi.detect_roi_3d(self.b_raw, self.method)
                roic = roib
                b_cropped = self.b_raw[
                    roic[0][0] : roic[0][1],
                    roic[1][0] : roic[1][1],
                    roic[2][0] : roic[2][1],
                ]
                a_cropped = None

            else:
                # Process both channels
                roia = roi.detect_roi_3d(self.a_raw, self.method)
                roib = roi.detect_roi_3d(self.b_raw, self.method)
                roic = roi.combine_rois(roia, roib)

                a_cropped = self.a_raw[
                    roic[0][0] : roic[0][1],
                    roic[1][0] : roic[1][1],
                    roic[2][0] : roic[2][1],
                ]
                b_cropped = self.b_raw[
                    roic[0][0] : roic[0][1],
                    roic[1][0] : roic[1][1],
                    roic[2][0] : roic[2][1],
                ]

            result = {
                "a_cropped": a_cropped,
                "b_cropped": b_cropped,
                "roi_coords": roic,
                "method": self.method,
                "process_single_channel": self.process_single_channel,
            }

            self.roi_detected.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))


class RoiDetectionWidget(QWidget):
    """Widget for ROI detection and cropping."""

    roi_applied = pyqtSignal(dict)

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

        # Layer selection
        layer_group = QGroupBox("Layer Selection")
        layer_layout = QFormLayout()

        self.layer_a_combo = QComboBox()
        self.layer_a_combo.addItem("Select Channel A layer...")
        self.layer_a_combo.currentTextChanged.connect(self.on_layer_selection_changed)

        self.layer_b_combo = QComboBox()
        self.layer_b_combo.addItem("Select Channel B layer...")
        self.layer_b_combo.currentTextChanged.connect(self.on_layer_selection_changed)

        layer_layout.addRow("Channel A:", self.layer_a_combo)
        layer_layout.addRow("Channel B:", self.layer_b_combo)
        layer_group.setLayout(layer_layout)

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
        self.method_combo.addItems(["otsu", "triangle"])
        self.method_combo.setCurrentText("otsu")
        self.method_combo.currentTextChanged.connect(self.on_method_changed)

        method_layout.addRow("Threshold Method:", self.method_combo)
        method_group.setLayout(method_layout)

        # Acquisition parameters (for reference and metadata)
        acq_group = QGroupBox("Acquisition Parameters")
        acq_layout = QFormLayout()

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

        acq_layout.addRow("Step Size:", self.step_size_spin)
        acq_layout.addRow("Pixel Size:", self.pixel_size_spin)
        acq_layout.addRow("Theta (angle):", self.theta_spin)
        acq_group.setLayout(acq_layout)





        # Detection button
        self.detect_button = QPushButton("Detect and Apply ROI")
        self.detect_button.clicked.connect(self.detect_roi)
        self.detect_button.setEnabled(False)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Status label
        self.status_label = QLabel("Select layers for ROI detection")

        # ROI info display
        self.roi_info_label = QLabel("")
        self.roi_info_label.setWordWrap(True)

        # Add widgets to layout
        layout.addWidget(layer_group)
        layout.addWidget(channel_group)
        layout.addWidget(method_group)
        layout.addWidget(acq_group)
        layout.addWidget(self.detect_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.roi_info_label)
        layout.addStretch()

        self.setLayout(layout)

        # Update layer lists when viewer layers change
        self.viewer.layers.events.inserted.connect(self.update_layer_lists)
        self.viewer.layers.events.removed.connect(self.update_layer_lists)

    def update_layer_lists(self, event=None):
        """Update the layer selection dropdowns."""
        # Store current selections
        current_a = self.layer_a_combo.currentText()
        current_b = self.layer_b_combo.currentText()

        # Clear and repopulate
        self.layer_a_combo.clear()
        self.layer_b_combo.clear()

        # Add placeholder items
        self.layer_a_combo.addItem("Select Channel A layer...")
        self.layer_b_combo.addItem("Select Channel B layer...")

        # Add available layers
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and layer.data is not None:
                self.layer_a_combo.addItem(layer.name)
                self.layer_b_combo.addItem(layer.name)

        # Restore selections if they still exist
        if current_a and current_a in [self.layer_a_combo.itemText(i) for i in range(self.layer_a_combo.count())]:
            self.layer_a_combo.setCurrentText(current_a)
        if current_b and current_b in [self.layer_b_combo.itemText(i) for i in range(self.layer_b_combo.count())]:
            self.layer_b_combo.setCurrentText(current_b)

        self.on_layer_selection_changed()

    def on_layer_selection_changed(self):
        """Handle layer selection changes."""
        a_selected = self.layer_a_combo.currentText() != "Select Channel A layer..."
        b_selected = self.layer_b_combo.currentText() != "Select Channel B layer..."

        # Update parameters from selected layers
        self._update_parameters_from_selected_layers()

        # Update button states
        has_data = a_selected or b_selected
        
        print(f"=== DEBUG: Layer selection changed ===")
        print(f"  a_selected: {a_selected} (text: '{self.layer_a_combo.currentText()}')")
        print(f"  b_selected: {b_selected} (text: '{self.layer_b_combo.currentText()}')")
        print(f"  has_data: {has_data}")
        print(f"  detect_button enabled: {has_data}")
        
        self.detect_button.setEnabled(has_data)

        # Update status
        if a_selected and b_selected:
            self.status_label.setText("Ready for ROI detection")
        elif a_selected:
            self.status_label.setText("Channel A selected - ready for ROI detection")
        elif b_selected:
            self.status_label.setText("Channel B selected - ready for ROI detection")
        else:
            self.status_label.setText("Select layers for ROI detection")

    def _update_parameters_from_selected_layers(self):
        """Update parameters from selected layer metadata."""
        # Check both selected layers for metadata
        layers_to_check = []
        
        if self.layer_a_combo.currentText() != "Select Channel A layer...":
            try:
                layer_a = self.viewer.layers[self.layer_a_combo.currentText()]
                layers_to_check.append(layer_a)
            except (KeyError, ValueError):
                pass

        if self.layer_b_combo.currentText() != "Select Channel B layer...":
            try:
                layer_b = self.viewer.layers[self.layer_b_combo.currentText()]
                layers_to_check.append(layer_b)
            except (KeyError, ValueError):
                pass

        # Try to get parameters from layer metadata
        for layer in layers_to_check:
            if hasattr(layer, 'metadata') and layer.metadata:
                metadata = layer.metadata
                
                if 'step_size' in metadata:
                    self.step_size_spin.setValue(metadata['step_size'])
                
                if 'pixel_size' in metadata:
                    self.pixel_size_spin.setValue(metadata['pixel_size'])
                
                if 'theta' in metadata:
                    theta_rad = metadata['theta']
                    theta_deg = theta_rad * 180 / math.pi
                    self.theta_spin.setValue(theta_deg)
                
                # Found parameters, no need to check other layers
                break

    def on_method_changed(self, method):
        """Handle method selection change."""
        # Update button states based on layer selection
        a_selected = self.layer_a_combo.currentText() != "Select Channel A layer..."
        b_selected = self.layer_b_combo.currentText() != "Select Channel B layer..."
        has_data = a_selected or b_selected
        
        print(f"=== DEBUG: Method changed to '{method}' ===")
        print(f"  a_selected: {a_selected} (text: '{self.layer_a_combo.currentText()}')")
        print(f"  b_selected: {b_selected} (text: '{self.layer_b_combo.currentText()}')")
        print(f"  has_data: {has_data}")
        
        self.detect_button.setEnabled(has_data)



    def set_input_data(self, data_dict):
        """Set input data from previous step (for backward compatibility)."""
        self.input_data = data_dict
        # Automatically update parameters from input data if available
        if data_dict:
            if "step_size" in data_dict:
                self.step_size_spin.setValue(data_dict["step_size"])
            
            if "pixel_size" in data_dict:
                self.pixel_size_spin.setValue(data_dict["pixel_size"])
            
            if "theta" in data_dict:
                theta_rad = data_dict["theta"]
                theta_deg = theta_rad * 180 / math.pi
                self.theta_spin.setValue(theta_deg)

    def detect_roi(self):
        """Detect ROI using selected method."""
        # Get selected layers
        a_layer_name = self.layer_a_combo.currentText()
        b_layer_name = self.layer_b_combo.currentText()
        
        if a_layer_name == "Select Channel A layer..." and b_layer_name == "Select Channel B layer...":
            QMessageBox.warning(self, "Error", "Please select at least one layer for ROI detection")
            return

        # Get data from selected layers
        a_raw = None
        b_raw = None

        if a_layer_name != "Select Channel A layer...":
            try:
                a_layer = self.viewer.layers[a_layer_name]
                a_raw = a_layer.data
            except (KeyError, ValueError) as e:
                QMessageBox.warning(self, "Error", f"Could not get data from layer {a_layer_name}: {e}")
                return

        if b_layer_name != "Select Channel B layer...":
            try:
                b_layer = self.viewer.layers[b_layer_name]
                b_raw = b_layer.data
            except (KeyError, ValueError) as e:
                QMessageBox.warning(self, "Error", f"Could not get data from layer {b_layer_name}: {e}")
                return

        method = self.method_combo.currentText()

        # Determine which channel to process
        process_single_channel = None
        if self.channel_a_radio.isChecked():
            process_single_channel = "a"
        elif self.channel_b_radio.isChecked():
            process_single_channel = "b"

        self.detect_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Detecting ROI...")

        # Create and start worker
        self.worker = RoiDetectionWorker(a_raw, b_raw, method, process_single_channel)
        self.worker.roi_detected.connect(self.on_roi_detected)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()





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
            "roi_coords": roi_coords,
            "method": method,
            "original_shape": data.shape,
        }

        # Add current spin box values to metadata
        metadata["step_size"] = self.step_size_spin.value()
        metadata["pixel_size"] = self.pixel_size_spin.value()
        metadata["theta"] = self.theta_spin.value() * math.pi / 180

        # Add new layer
        self.viewer.add_image(data, name=name, metadata=metadata)

    def on_roi_detected(self, result):
        """Handle successful ROI detection."""
        a_cropped = result["a_cropped"]
        b_cropped = result["b_cropped"]
        roi_coords = result["roi_coords"]

        # Add cropped data as new layers (only for processed channels)
        if a_cropped is not None:
            source_name = self.layer_a_combo.currentText()
            output_name = f"{source_name}_cropped"
            self._add_cropped_layer(
                a_cropped, output_name, roi_coords, result["method"]
            )
        if b_cropped is not None:
            source_name = self.layer_b_combo.currentText()
            output_name = f"{source_name}_cropped"
            self._add_cropped_layer(
                b_cropped, output_name, roi_coords, result["method"]
            )

        # Update status and info
        if a_cropped is not None and b_cropped is not None:
            status_text = (
                f"ROI applied successfully! A: {a_cropped.shape}, B: {b_cropped.shape}"
            )
        elif a_cropped is not None:
            status_text = f"ROI applied successfully! A: {a_cropped.shape}"
        else:
            status_text = f"ROI applied successfully! B: {b_cropped.shape}"

        self.status_label.setText(status_text)
        self._update_roi_info(roi_coords)

        self.detect_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Emit signal with cropped data
        output_data = result.copy()
        # Add current parameters to output data
        output_data["step_size"] = self.step_size_spin.value()
        output_data["pixel_size"] = self.pixel_size_spin.value()
        output_data["theta"] = self.theta_spin.value() * math.pi / 180
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
