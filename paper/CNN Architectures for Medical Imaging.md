# [Medical ML] CNN Architectures for Medical Imaging: From ResNet to EfficientNet

# CNN Architectures for Medical Imaging: From ResNet to EfficientNet

*By Oleh Ivchenko | February 8, 2026*

Convolutional Neural Networks (CNNs) have fundamentally transformed medical image analysis, evolving from simple feature extractors to sophisticated architectures capable of matching or exceeding radiologist-level performance. This article provides a comprehensive technical deep-dive into the CNN architectures that power modern medical AI systems, examining their design principles, clinical applications, and performance characteristics.

## The Evolution of Medical Imaging CNNs

The journey from basic convolution operations to today's state-of-the-art architectures represents one of the most significant advances in computational medicine. Understanding this evolution is crucial for implementing effective diagnostic AI systems.


graph LR
    A[LeNet 1998] --&gt; B[AlexNet 2012]
    B --&gt; C[VGG ResNet 2014]
    C --&gt; D[DenseNet 2016]
    D --&gt; E[EfficientNet 2019]
    E --&gt; F[Hybrid 2024]


## Core Architecture Families

### ResNet: The Residual Revolution

ResNet (Residual Networks) introduced skip connections that allow gradients to flow directly through the network, enabling training of extremely deep architectures without vanishing gradient problems.


graph TD
    A[Input x] --&gt; B[Conv Layer 1]
    B --&gt; C[BatchNorm]
    C --&gt; D[Conv Layer 2]
    A --&gt; E[Skip Connection]
    E --&gt; F[Add and Output]


**Mathematical Formulation:**

For a traditional CNN layer: **H(x) = F(x)**

For ResNet: **H(x) = F(x) + x**

This simple modification has profound implications. The network learns the *residual* F(x) = H(x) - x, which is easier to optimize when the optimal mapping is close to identity.

**Clinical Applications:**
- **Pulmonary nodule detection:** ResNet-50 achieves 94.2% sensitivity in lung CT analysis
- **Intracranial hemorrhage classification:** Deep ResNets excel at fine-grained CT detail extraction
- **Bone fracture detection:** Residual connections preserve subtle skeletal features




ResNet Variant
Layers
Parameters
Top-1 Accuracy (ImageNet)
Medical Imaging Use Case




ResNet-18
18
11.7M
69.8%
Quick screening, mobile deployment


ResNet-34
34
21.8M
73.3%
Balanced performance/efficiency


ResNet-50
50
25.6M
76.1%
Most common medical imaging baseline


ResNet-101
101
44.5M
77.4%
Complex multi-class classification


ResNet-152
152
60.2M
78.3%
Research, maximum feature extraction




### DenseNet: Maximizing Feature Reuse

DenseNet (Densely Connected Networks) takes connectivity to the extreme: each layer receives inputs from *all* preceding layers and passes its feature maps to *all* subsequent layers.


graph LR
    A[Layer 0] --&gt; B[Layer 1]
    A --&gt; C[Layer 2]
    A --&gt; D[Layer 3]
    B --&gt; C
    B --&gt; D
    C --&gt; D


**Key Innovation:** Feature concatenation instead of summation

- ResNet: **H(x) = F(x) + x** (addition)
- DenseNet: **H(x) = [x₀, x₁, ..., xₗ₋₁]** (concatenation)

This design provides:
- **3× parameter reduction** compared to ResNet with equivalent performance
- **Improved gradient flow** through direct connections
- **Feature reuse** reducing redundancy
- **Implicit deep supervision** from multiple paths

**CheXNet Breakthrough:** DenseNet-121 trained on 112,120 chest X-rays achieved radiologist-level performance in pneumonia detection, demonstrating the architecture's effectiveness for medical imaging.




Medical Application
DenseNet Variant
Performance (AUC)
Dataset




Chest X-ray multi-label
DenseNet-121
0.841
ChestX-ray14


Pancreatic cyst classification
DenseNet-169
0.89
Institutional CT


Thymoma staging
DenseNet-201
0.92
Masaoka-Koga


COVID-19 detection
DenseNet-121
0.96
COVIDx CT




### Inception: Multi-Scale Feature Extraction

The Inception architecture captures information at multiple scales simultaneously by applying different filter sizes in parallel within each module.


graph TD
    A[Previous Layer] --&gt; B[1x1 Conv]
    A --&gt; C[3x3 Conv]
    A --&gt; D[5x5 Conv]
    A --&gt; E[MaxPool]
    B --&gt; F[Concatenate]
    C --&gt; F


**Design Principles:**
1. **Multi-scale processing:** Parallel convolutions with 1×1, 3×3, and 5×5 filters
2. **Dimensionality reduction:** 1×1 convolutions before larger filters reduce computation
3. **Sparse connections:** Computationally efficient approximation of optimal sparse structure

**Evolution:**
- **Inception v1 (GoogLeNet):** Original architecture, 22 layers
- **Inception v2/v3:** Batch normalization, factorized convolutions
- **Inception v4:** Streamlined design, uniform reduction blocks
- **Inception-ResNet:** Combines Inception modules with residual connections

**Medical Imaging Applications:**
- Lung cancer staging (multi-scale tumor features)
- Kidney cancer classification (varied lesion sizes)
- Osteomeatal complex inflammation detection

### EfficientNet: Optimal Scaling

EfficientNet revolutionizes architecture design by introducing *compound scaling* — simultaneously scaling network depth, width, and resolution using a principled approach.


graph LR
    A[Base Model] --&gt; B[Compound Scaling]
    B --&gt; C[Depth]
    B --&gt; D[Width]
    B --&gt; E[Resolution]
    E --&gt; F[Scaled Model]


**Compound Scaling Formula:**

- depth: d = α^φ
- width: w = β^φ  
- resolution: r = γ^φ

Where α · β² · γ² ≈ 2 (resource constraint) and φ controls overall scaling.




Model
Input Size
Parameters
FLOPs
Top-1 Acc




EfficientNet-B0
224×224
5.3M
0.39B
77.1%


EfficientNet-B3
300×300
12M
1.8B
81.6%


EfficientNet-B4
380×380
19M
4.2B
82.9%


EfficientNet-B7
600×600
66M
37B
84.3%




**Medical Imaging Advantages:**
- **Optimal resource utilization:** Best accuracy-per-FLOP ratio
- **Scalable deployment:** B0 for edge devices, B7 for research
- **Small lesion detection:** Higher resolution variants excel at subtle findings

### MobileNet: Edge Deployment

MobileNet enables deployment of medical AI on resource-constrained devices through depthwise separable convolutions.


graph TD
    A[Input Image] --&gt; B[Depthwise Conv]
    B --&gt; C[Pointwise Conv]
    C --&gt; D[Output Features]
    D --&gt; E[Classification]


**Computational Savings:**

Standard convolution cost: **DK² × M × N × DF²**

Depthwise separable cost: **DK² × M × DF² + M × N × DF²**

Reduction ratio: **1/N + 1/DK²** (typically 8-9× fewer operations)

**Mobile Medical AI Applications:**
- Point-of-care skin lesion screening
- Portable ultrasound analysis
- Field-deployable chest X-ray triage

## Attention Mechanisms in Medical CNNs

### Squeeze-and-Excitation Networks (SE-Net)

SE-Net introduces channel attention by explicitly modeling interdependencies between channels, allowing the network to emphasize informative features.


graph LR
    A[Feature Map] --&gt; B[Global Pool]
    B --&gt; C[FC Layers]
    C --&gt; D[Sigmoid]
    D --&gt; E[Scale Features]
    E --&gt; F[Output]


**Clinical Impact:**
- **Pulmonary nodule detection:** SE-ResNet achieves 12% sensitivity improvement
- **Spatial attention focus:** Concentrates on discriminative anatomical regions
- **3D extension:** SE-3D networks for volumetric CT/MRI analysis

### Convolutional Block Attention Module (CBAM)

CBAM applies both channel and spatial attention sequentially, providing comprehensive feature refinement.


graph LR
    A[Input F] --&gt; B[Channel Attention]
    B --&gt; C[Refined F]
    C --&gt; D[Spatial Attention]
    D --&gt; E[Output F]


## U-Net: The Segmentation Standard

While not strictly a classification architecture, U-Net deserves special mention for its dominance in medical image segmentation tasks.


graph TD
    A[Encoder 64] --&gt; B[Encoder 128]
    B --&gt; C[Encoder 256]
    C --&gt; D[Bottom 512]
    D --&gt; E[Decoder 256]
    E --&gt; F[Decoder 128]


**U-Net Variants for Medical Imaging:**




Variant
Key Innovation
Application




3D U-Net
Volumetric convolutions
CT/MRI organ segmentation


Attention U-Net
Attention gates in skip connections
Improved boundary detection


U-Net++
Nested dense skip pathways
Multi-scale feature fusion


nnU-Net
Self-configuring framework
Automatic architecture selection


TransUNet
Transformer encoder + U-Net decoder
Long-range dependency modeling




## Performance Benchmarks on Medical Datasets

### MedMNIST Benchmark Results (2024-2025)




Architecture
DermaMNIST
BloodMNIST
PathMNIST
OrganAMNIST
Avg AUC




ResNet-18
0.912
0.987
0.978
0.996
0.968


ResNet-50
0.917
0.991
0.982
0.997
0.972


DenseNet-121
0.921
0.993
0.985
0.998
0.974


EfficientNet-B0
0.915
0.989
0.980
0.996
0.970


EfficientNet-B4
0.928
0.994
0.988
0.998
0.977


VGG-16
0.909
0.985
0.976
0.995
0.966


MedNet (2025)
0.932
0.995
0.989
0.998
0.979




**Key Finding (2025):** CNNs, particularly DenseNet-121 and VGG-16, consistently outperform Vision Transformers in end-to-end training on medical imaging datasets when sufficient data is available, highlighting the continued relevance of CNN architectures.

## Architecture Selection Guide


graph TD
    A[Select Architecture] --&gt; B{Deploy Target}
    B --&gt; C[Cloud Server]
    B --&gt; D[Mobile Edge]
    C --&gt; E[ResNet DenseNet]
    D --&gt; F[MobileNet EfficientNet]


## Practical Recommendations for Clinical Implementation

### For Routine Clinical Use
**Recommended:** Xception, MobileNet, EfficientNet-B0/B1
- Optimal accuracy-efficiency tradeoff
- Real-time inference capability
- Edge deployment feasible

### For Spatial Discrimination Tasks
**Recommended:** SE-Net, Inception-ResNet, CBAM-enhanced models
- Lung nodule localization
- Hemorrhage detection
- Tumor boundary delineation

### For Multi-Structure Classification
**Recommended:** DenseNet, ResNet ensembles
- Robust feature extraction
- Multiple pathology detection
- Large-scale screening programs

### For Research and Maximum Performance
**Recommended:** EfficientNet-B4+, ResNet-152
- State-of-the-art accuracy
- Comprehensive feature learning
- Publication-grade results

## Conclusion

The landscape of CNN architectures for medical imaging continues to evolve rapidly. While newer transformer-based approaches show promise, CNNs remain the backbone of clinical AI deployment due to their:

- **Proven clinical efficacy** across multiple FDA-approved devices
- **Computational efficiency** enabling real-time inference
- **Interpretability** through established visualization techniques
- **Transfer learning capability** with extensive pretrained weights

For Ukrainian healthcare implementations, the path forward involves:
1. Starting with established architectures (ResNet-50, DenseNet-121)
2. Applying transfer learning from ImageNet or RadImageNet
3. Fine-tuning with local patient demographics
4. Deploying lightweight variants for point-of-care applications

The next article in this series will explore Vision Transformers in Radiology, examining how attention mechanisms are pushing the boundaries of medical image analysis.

---

## References

1. He K., et al. "Deep residual learning for image recognition." CVPR 2016.
2. Huang G., et al. "Densely connected convolutional networks." CVPR 2017.
3. Tan M., Le Q.V. "EfficientNet: Rethinking model scaling for convolutional neural networks." ICML 2019.
4. Szegedy C., et al. "Going deeper with convolutions." CVPR 2015.
5. Hu J., et al. "Squeeze-and-excitation networks." CVPR 2018.
6. Woo S., et al. "CBAM: Convolutional block attention module." ECCV 2018.
7. Ronneberger O., et al. "U-Net: Convolutional networks for biomedical image segmentation." MICCAI 2015.
8. Howard A.G., et al. "MobileNets: Efficient convolutional neural networks for mobile vision applications." arXiv 2017.
9. "MedNet: A lightweight attention-augmented CNN for medical image classification." Scientific Reports 2025.
10. "Deep learning models for CT image classification: A comprehensive literature review." PMC 2025.
11. "A review of convolutional neural network based methods for medical image classification." Computers in Biology and Medicine 2024.
12. "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data 2023.

---


  
  
    CNN architectures for medical imaging analysis
  




*This article is part of the Medical ML for Diagnosis research series exploring machine learning applications in Ukrainian healthcare. The series aims to provide a comprehensive framework for implementing AI-assisted diagnostic systems.*
