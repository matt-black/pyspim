"""
Main widget for the diSPIM processing pipeline.

This widget provides a tabbed interface for all processing steps:
0. Remote Connection (optional)
1. ROI Detection
2. Registration
3. Deconvolution
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QWidget

from . import HAS_PYSPIM
from ._deconvolution import DeconvolutionWidget
from ._remote_client import RemoteClient
from ._remote_connection import RemoteConnectionWidget
from ._registration import RegistrationWidget
from ._roi_detection import RoiDetectionWidget


class DispimPipelineWidget(QWidget):
    """Main widget containing all diSPIM processing steps."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.remote_client = RemoteClient()
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface with tabs for each processing step."""
        layout = QVBoxLayout()

        # Use Preferred horizontally so napari gives a reasonable initial width,
        # but Ignored vertically so the dock height is not forced to grow.
        # The scroll area below will handle vertical overflow instead.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Create Remote Connection widget
        self.remote_connection = RemoteConnectionWidget(self.remote_client)

        # Create individual step widgets, passing HAS_PYSPIM to control local mode
        self.roi_detection = RoiDetectionWidget(self.viewer, self.remote_client, has_pyspim=HAS_PYSPIM)
        self.registration = RegistrationWidget(self.viewer, self.remote_client, has_pyspim=HAS_PYSPIM)
        self.deconvolution = DeconvolutionWidget(self.viewer, self.remote_client, has_pyspim=HAS_PYSPIM)

        # Add tabs
        self.tab_widget.addTab(self.remote_connection, "0. Remote Connection")
        self.tab_widget.addTab(self.roi_detection, "1. ROI Detection")
        self.tab_widget.addTab(self.registration, "2. Registration")
        self.tab_widget.addTab(self.deconvolution, "3. Deconvolution")

        # Connect signals for data flow between steps
        self._connect_signals()

        # Wrap the tab widget in a scroll area so the napari window size
        # stays fixed and a scrollbar appears when content overflows.
        scroll = QScrollArea()
        scroll.setWidget(self.tab_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        layout.addWidget(scroll)
        self.setLayout(layout)

    def _connect_signals(self):
        """Connect signals between widgets for data flow."""
        # ROI detection -> Registration
        self.roi_detection.roi_applied.connect(self.registration.set_input_data)

        # Registration -> Deconvolution
        self.registration.registered.connect(self.deconvolution.set_input_data)
