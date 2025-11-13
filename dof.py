import math


sensor_width_mm = 6.3
aperture_N = 2.8
subject_distance_mm = 1000     
focal_length_mm = 8.0

# Known object and pixel measurement
real_object_width_mm = 100.0
pixel_x1 = 200
pixel_x2 = 600
pixel_width = pixel_x2 - pixel_x1

# Pixel resolution
pixel_resolution = real_object_width_mm / pixel_width
circle_of_confusion = pixel_resolution  # CoC = pixel pitch

print(f"Pixel Resolution: {pixel_resolution:.4f} mm/pixel")

# Hyperfocal distance
H = (focal_length_mm**2) / (aperture_N * circle_of_confusion)

# Near and far focus distances
D = subject_distance_mm

D_near = (H * D) / (H + (D - focal_length_mm))
D_far = (H * D) / (H - (D - focal_length_mm))

print(f"Near Focus Distance: {D_near:.2f} mm")
print(f"Far Focus Distance:  {D_far}")
print(f"Total DOF:           {DOF}")
