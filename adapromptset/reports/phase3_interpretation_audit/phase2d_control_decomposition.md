# Phase 2D Control Decomposition

C2 is interpreted as an alternative-reference control, not a true negative control, because it uses another GlaS gland reference from the frozen support manifest.

| Segmenter | Family | Priv-C1 | Priv-C2 | C1-C2 | Win vs C1 | Win vs C2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SAM-B | P1_largest_component_centroid | `-0.042701` | `-0.095940` | `-0.053239` | `0.433333` | `0.366667` |
| SAM-B | P2_fixed_multi_centroid_one_group | `-0.083978` | `-0.043278` | `0.040700` | `0.400000` | `0.500000` |
| SAM-B | P3_component_grouped_centroids | `-0.159645` | `-0.156230` | `0.003415` | `0.433333` | `0.433333` |
| SAM-B | P4_union_mask_bounding_box | `0.044447` | `0.064274` | `0.019826` | `0.533333` | `0.633333` |
| SAM-B | P5_positive_plus_controlled_negative | `-0.200921` | `-0.098899` | `0.102022` | `0.366667` | `0.400000` |
| SAM-L | P1_largest_component_centroid | `0.082163` | `-0.058360` | `-0.140524` | `0.600000` | `0.466667` |
| SAM-L | P2_fixed_multi_centroid_one_group | `0.001017` | `0.012036` | `0.011019` | `0.566667` | `0.533333` |
| SAM-L | P3_component_grouped_centroids | `-0.074410` | `-0.119408` | `-0.044998` | `0.400000` | `0.366667` |
| SAM-L | P4_union_mask_bounding_box | `0.039224` | `0.094778` | `0.055553` | `0.500000` | `0.666667` |
| SAM-L | P5_positive_plus_controlled_negative | `0.049075` | `-0.015670` | `-0.064745` | `0.600000` | `0.466667` |
