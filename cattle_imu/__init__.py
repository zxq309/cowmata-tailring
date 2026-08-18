"""Cattle tail-ring IMU research pipeline."""

from .io import MAG_RAW_TO_IMU, ImuSession, load_v2_json, rotate_mag_raw_to_imu

__all__ = [
    "MAG_RAW_TO_IMU",
    "ImuSession",
    "load_v2_json",
    "rotate_mag_raw_to_imu",
]

__version__ = "0.1.0"
