"""
diSPIM Processing Pipeline - A napari plugin for processing dual-view SPIM data.
"""

# Import the main widget for npe2 discovery
from ._main_widget import DispimPipelineWidget
from ._remote_client import RemoteClient
from ._remote_connection import RemoteConnectionWidget
from ._sftp_browser import SftpBrowserDialog

__all__ = [
    "DispimPipelineWidget",
    "RemoteClient",
    "RemoteConnectionWidget",
    "SftpBrowserDialog",
]
