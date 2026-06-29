# HexMFG: Causal Voronoi Isotropic Convolution with Adaptive Frequency Gating for Efficient Precipitation Nowcasting
HexMFG is the generator backbone of the proposed HexM lightweight GAN framework, specially designed to address high-frequency echo dissipation and temporal error accumulation in radar precipitation nowcasting. Combined with the matching HexMF-DNet discriminator, the whole model realizes accurate, low-overhead short-term rainfall prediction for HPC operational systems.

## Model Architecture
The overall HexMFG framework consists of two core components: the generator HexMF-GNet and discriminator HexMF-DNet, as illustrated in Figure 2. The generator maps historical radar echo sequences to future forecast frames, while the discriminator distinguishes authentic radar data from model-generated outputs for adversarial optimization.

### HexMF-GNet: Generator
HexMF-GNet adopts an encoder-translator-decoder symmetric cascade architecture, composed of HMS Encoder, HFF network, MF-Translator and HMS Decoder four core modules. It relies on the proposed Probabilistic Causal Isotropic Convolution (PCIC) and dual adaptive frequency gating to retain fine radar textures without massive downsampling.

- **HMS Encoder**: Four-level hierarchical feature extraction pipeline. Each layer embeds HFF networks to extract multi-scale spatiotemporal features; stride-2 convolution downsamples feature maps between stages to expand receptive fields progressively.
- **HFF Network (Honeycomb Feature Fusion)**: Core feature extraction unit built upon three parallel PCIC branches with distinct receptive fields (3×3 / 5×5 / 7×7). Equipped with AG-2 spatial adaptive gating, it fuses convective, mesoscale and synoptic rainfall features via residual connection, suppressing high-frequency attenuation at feature extraction stage.
- **MF-Translator (Multi-Frequency Translator)**: Bottleneck fusion module with dual parallel paths. The spatial path extracts multi-scale features via PCIC and depthwise separable convolution; the frequency path decomposes echoes with Gaussian/Laplacian/Sobel filters and AG-1 frequency gating. GBCG bottleneck fusion unifies two branches to produce compact multi-frequency bottleneck features with minimal computation cost.
- **HMS Decoder**: Symmetric upsampling structure matching the encoder. Bilinear interpolation recovers spatial resolution, and skip connections transmit shallow fine-grained features. HFF + CBAM refine fused features to reconstruct complete precipitation echo morphology for final prediction output.

#### Core Operator: Probabilistic Causal Isotropic Convolution (PCIC)
As the basic convolution unit of HexMF-GNet, PCIC constructs data-adaptive Voronoi receptive fields via weighted power diagrams to eliminate the directional anisotropy of standard square convolution. Learnable temporal causal masks are embedded inside the operator to block future information leakage, effectively reducing long-range forecast error accumulation and preserving heavy rainfall core intensity and edge details.

### HexMF-DNet: Discriminator
HexMF-DNet is a lightweight matching discriminator cooperating with HexMF-GNet, consisting of three cascaded lightweight modules to constrain physically consistent precipitation structures during adversarial training:
1. **MSST (Multi-Scale Spatiotemporal Encoding)**
Takes concatenated input of historical + forecast radar frames. Lightweight 3D pooling and convolution stacks extract multi-resolution texture and motion features with low memory overhead.

2. **PAF (Pixel Attention-oriented Fusion)**
Dual-branch feature enhancement module integrated with the PASTA attention mechanism. The shallow branch highlights small convective edge features; the deep branch captures global rainfall circulation patterns. PASTA adaptively boosts feature weights of heavy precipitation regions and suppresses useless background noise.

3. **PSD (Precipitation Structure Discriminator)**
Fuses local texture and global attention features, and introduces gradient structural constraints to filter unrealistic echo artifacts such as isolated intense spots. It outputs authenticity scores and structural loss to regularize the generator’s output morphology.

### PASTA Attention Mechanism
PASTA (Precipitation-Aware Spatio-Temporal Attention) is the shared lightweight attention unit of HexMF-GNet and HexMF-DNet. Tailored for radar echo characteristics, it builds joint channel-spatial weighting and explicitly models continuous spatiotemporal evolution of rainfall systems. Different from generic attention modules, it prioritizes feature regions with strong convection, improving the temporal smoothness and structural integrity of forecast sequences without excessive parameter growth.

# HexMFG
Precipitation Nowcasting, Generative Adversarial Network, Adaptive Frequency Gating, Probabilistic Causal Isotropic Convolution
# Clone the repository
git clone https://github.com/JustlikeCatnip/HexMFG
cd HexMFG

# Create conda environment
conda create -n hexmfg python=3.8
conda activate hexmfg

# Install dependencies
pip install -r requirements.txt

#run
python train_precip.py

#test
python test_precip.py

#dataset
We adopt the 2020–2025 gauge-calibrated KNMI radar precipitation HDF5 dataset with 5-min temporal resolution and 64×64 spatial grids, which records temperate marine rainfall characteristics across the Netherlands and contains 23,329 spatiotemporal samples split into training and test sets.
