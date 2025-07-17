"""
Main widget for the diSPIM processing pipeline.

This widget provides a tabbed interface for all processing steps:
1. Data Loading
2. ROI Detection  
3. Deskewing
4. Registration
5. Deconvolution
"""

from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from ._data_loader import DataLoaderWidget
from ._roi_detection import RoiDetectionWidget
from ._deskewing import DeskewingWidget
from ._registration import RegistrationWidget
from ._deconvolution import DeconvolutionWidget


class DispimPipelineWidget(QWidget):
    """Main widget containing all diSPIM processing steps."""
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface with tabs for each processing step."""
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create individual step widgets
        self.data_loader = DataLoaderWidget(self.viewer)
        self.roi_detection = RoiDetectionWidget(self.viewer)
        self.deskewing = DeskewingWidget(self.viewer)
        self.registration = RegistrationWidget(self.viewer)
        self.deconvolution = DeconvolutionWidget(self.viewer)
        
        # Add tabs
        self.tab_widget.addTab(self.data_loader, "1. Data Loading")
        self.tab_widget.addTab(self.roi_detection, "2. ROI Detection")
        self.tab_widget.addTab(self.deskewing, "3. Deskewing")
        self.tab_widget.addTab(self.registration, "4. Registration")
        self.tab_widget.addTab(self.deconvolution, "5. Deconvolution")
        
        # Connect signals for data flow between steps
        self._connect_signals()
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
    def _connect_signals(self):
        """Connect signals between widgets for data flow."""
        # Data loader -> ROI detection
        self.data_loader.data_loaded.connect(self.roi_detection.set_input_data)
        
        # ROI detection -> Deskewing
        self.roi_detection.roi_applied.connect(self.deskewing.set_input_data)
        
        # Deskewing -> Registration
        self.deskewing.deskewed.connect(self.registration.set_input_data)
        
        # Registration -> Deconvolution
        self.registration.registered.connect(self.deconvolution.set_input_data) 