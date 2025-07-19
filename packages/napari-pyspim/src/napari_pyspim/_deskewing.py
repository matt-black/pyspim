"""
Deskewing widget for deskewing dual-view SPIM data.
"""

import math

import numpy as np
from PyQt5.QtCore import pyqtSignal
from qtpy.QtCore import QThread
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Lazy imports to avoid CUDA compilation at module level
# from pyspim import deskew as dsk
# from pyspim import roi


class DeskewingWorker(QThread):
    """Worker thread for deskewing."""

    deskewed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        a_raw,
        b_raw,
        pixel_size,
        step_size,
        theta,
        method="orthogonal",
        recrop=False,
        process_single_channel=None,
    ):
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
            # Lazy imports to avoid CUDA compilation at module level
            from pyspim import deskew as dsk
            from pyspim import roi

            # Calculate step sizes
            step_size_lat = self.step_size / math.cos(self.theta)

            if self.process_single_channel == "a":
                # Process only channel A
                a_dsk = dsk.deskew_stage_scan(
                    self.a_raw, self.pixel_size, step_size_lat, 1, method=self.method
                )
                b_dsk = None

            elif self.process_single_channel == "b":
                # Process only channel B
                b_dsk = dsk.deskew_stage_scan(
                    self.b_raw, self.pixel_size, step_size_lat, -1, method=self.method
                )
                a_dsk = None

            else:
                # Process both channels
                # Deskew head A (direction = 1)
                a_dsk = dsk.deskew_stage_scan(
                    self.a_raw, self.pixel_size, step_size_lat, 1, method=self.method
                )

                # Deskew head B (direction = -1)
                b_dsk = dsk.deskew_stage_scan(
                    self.b_raw, self.pixel_size, step_size_lat, -1, method=self.method
                )

            # Optional re-cropping
            if self.recrop:
                if a_dsk is not None:
                    roia = roi.detect_roi_3d(a_dsk, "triangle")
                    a_dsk = a_dsk[
                        roia[0][0] : roia[0][1],
                        roia[1][0] : roia[1][1],
                        roia[2][0] : roia[2][1],
                    ].astype(np.float32)

                if b_dsk is not None:
                    roib = roi.detect_roi_3d(b_dsk, "triangle")
                    b_dsk = b_dsk[
                        roib[0][0] : roib[0][1],
                        roib[1][0] : roib[1][1],
                        roib[2][0] : roib[2][1],
                    ].astype(np.float32)
            else:
                if a_dsk is not None:
                    a_dsk = a_dsk.astype(np.float32)
                if b_dsk is not None:
                    b_dsk = b_dsk.astype(np.float32)

            result = {
                "a_deskewed": a_dsk,
                "b_deskewed": b_dsk,
                "method": self.method,
                "recrop": self.recrop,
                "step_size_lat": step_size_lat,
                "process_single_channel": self.process_single_channel,
            }

            self.deskewed.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))


class DeskewingWidget(QWidget):
    """Widget for deskewing dual-view data."""

    deskewed = pyqtSignal(dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.worker = None
        self.input_data = None
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

        # Deskewing parameters
        params_group = QGroupBox("Deskewing Parameters")
        params_layout = QFormLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["orthogonal"])
        self.method_combo.setCurrentText("orthogonal")

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

        self.recrop_check = QCheckBox("Re-crop after deskewing")
        self.recrop_check.setChecked(False)

        params_layout.addRow("Method:", self.method_combo)
        params_layout.addRow("Step Size:", self.step_size_spin)
        params_layout.addRow("Pixel Size:", self.pixel_size_spin)
        params_layout.addRow("Theta (angle):", self.theta_spin)
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
        self.status_label = QLabel("Select layers to deskew")

        # Results info
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)

        # Add widgets to layout
        layout.addWidget(layer_group)
        layout.addWidget(params_group)
        layout.addWidget(self.deskew_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.results_label)
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

        # Enable deskew button if at least one layer is selected
        self.deskew_button.setEnabled(a_selected or b_selected)

        # Update parameters from selected layers
        self._update_parameters_from_selected_layers()

        # Update status
        if a_selected and b_selected:
            self.status_label.setText("Ready to deskew both channels")
        elif a_selected:
            self.status_label.setText("Ready to deskew Channel A only")
        elif b_selected:
            self.status_label.setText("Ready to deskew Channel B only")
        else:
            self.status_label.setText("Select layers to deskew")

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

    def deskew_data(self):
        """Deskew the data using background worker."""
        # Get selected layers
        a_layer_name = self.layer_a_combo.currentText()
        b_layer_name = self.layer_b_combo.currentText()
        
        if a_layer_name == "Select Channel A layer..." and b_layer_name == "Select Channel B layer...":
            QMessageBox.warning(self, "Error", "Please select at least one layer to deskew")
            return

        # Get data from selected layers
        a_raw = None
        b_raw = None
        process_single_channel = None

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

        # Determine processing mode
        if a_raw is not None and b_raw is not None:
            process_single_channel = None  # Process both
        elif a_raw is not None:
            process_single_channel = "a"
        elif b_raw is not None:
            process_single_channel = "b"

        # Get parameters from spin boxes
        step_size = self.step_size_spin.value()
        pixel_size = self.pixel_size_spin.value()
        theta_deg = self.theta_spin.value()
        theta_rad = theta_deg * math.pi / 180
        method = self.method_combo.currentText()
        recrop = self.recrop_check.isChecked()

        self.deskew_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Deskewing data...")

        # Create and start worker
        self.worker = DeskewingWorker(
            a_raw,
            b_raw,
            pixel_size,
            step_size,
            theta_rad,
            method,
            recrop,
            process_single_channel,
        )
        self.worker.deskewed.connect(self.on_deskewed)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def on_deskewed(self, result):
        """Handle successful deskewing."""
        a_deskewed = result["a_deskewed"]
        b_deskewed = result["b_deskewed"]

        # Add deskewed data as new layers (only for processed channels)
        if a_deskewed is not None:
            source_name = self.layer_a_combo.currentText()
            output_name = f"{source_name}_deskewed"
            
            # Remove old layer if it exists
            try:
                layer = self.viewer.layers[output_name]
                self.viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass
            
            self.viewer.add_image(
                a_deskewed,
                name=output_name,
                metadata={
                    "method": result["method"],
                    "recrop": result["recrop"],
                    "step_size_lat": result["step_size_lat"],
                    "step_size": self.step_size_spin.value(),
                    "pixel_size": self.pixel_size_spin.value(),
                    "theta": self.theta_spin.value() * math.pi / 180,
                },
            )

        if b_deskewed is not None:
            source_name = self.layer_b_combo.currentText()
            output_name = f"{source_name}_deskewed"
            
            # Remove old layer if it exists
            try:
                layer = self.viewer.layers[output_name]
                self.viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass
            
            self.viewer.add_image(
                b_deskewed,
                name=output_name,
                metadata={
                    "method": result["method"],
                    "recrop": result["recrop"],
                    "step_size_lat": result["step_size_lat"],
                    "step_size": self.step_size_spin.value(),
                    "pixel_size": self.pixel_size_spin.value(),
                    "theta": self.theta_spin.value() * math.pi / 180,
                },
            )

        # Update status and results
        if a_deskewed is not None and b_deskewed is not None:
            status_text = (
                f"Deskewing completed! A: {a_deskewed.shape}, B: {b_deskewed.shape}"
            )
            results_text = f"""
            <b>Deskewing Results:</b><br>
            Method: {result["method"]}<br>
            Re-cropped: {result["recrop"]}<br>
            Lateral step size: {result["step_size_lat"]:.3f} μm<br>
            A shape: {a_deskewed.shape}<br>
            B shape: {b_deskewed.shape}
            """
        elif a_deskewed is not None:
            status_text = f"Deskewing completed! A: {a_deskewed.shape}"
            results_text = f"""
            <b>Deskewing Results:</b><br>
            Method: {result["method"]}<br>
            Re-cropped: {result["recrop"]}<br>
            Lateral step size: {result["step_size_lat"]:.3f} μm<br>
            A shape: {a_deskewed.shape}<br>
            B: Not processed
            """
        else:
            status_text = f"Deskewing completed! B: {b_deskewed.shape}"
            results_text = f"""
            <b>Deskewing Results:</b><br>
            Method: {result["method"]}<br>
            Re-cropped: {result["recrop"]}<br>
            Lateral step size: {result["step_size_lat"]:.3f} μm<br>
            A: Not processed<br>
            B shape: {b_deskewed.shape}
            """

        self.status_label.setText(status_text)
        self.results_label.setText(results_text)
        self.deskew_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Emit signal with deskewed data
        output_data = {
            "a_deskewed": a_deskewed,
            "b_deskewed": b_deskewed,
            "method": result["method"],
            "recrop": result["recrop"],
            "step_size_lat": result["step_size_lat"],
            "step_size": self.step_size_spin.value(),
            "pixel_size": self.pixel_size_spin.value(),
            "theta": self.theta_spin.value() * math.pi / 180,
        }
        self.deskewed.emit(output_data)

    def on_error(self, error_msg):
        """Handle deskewing error."""
        QMessageBox.critical(self, "Error", f"Deskewing failed: {error_msg}")
        self.status_label.setText("Error during deskewing")
        self.deskew_button.setEnabled(True)
        self.progress_bar.setVisible(False)
