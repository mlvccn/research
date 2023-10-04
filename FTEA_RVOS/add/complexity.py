from add.spatial_decoder import FPNSpatialDecoderV5
from models.segmentation import FPNSpatialDecoder

m = FPNSpatialDecoderV5(context_dim=256, fpn_dims=[192, 96])
print(m.num_parameters())
m1 = FPNSpatialDecoder(context_dim=256, fpn_dims=[192, 96])
print(m1.num_parameters())
