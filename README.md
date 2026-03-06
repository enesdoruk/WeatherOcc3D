# VLM-Assisted Adverse Weather Aware 3D Semantic Occupancy Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2601.14448-b31b1b.svg)](https://arxiv.org/abs/2601.14448)

**WeatherOcc3D** is a robust multi-modal 3D semantic occupancy prediction framework for autonomous driving. By leveraging a pre-trained CLIP text encoder, the model dynamically adjusts the trust between camera and LiDAR inputs based on real-time environmental conditions (e.g., shifting reliance to LiDAR during a rainy night).

![WeatherOcc3D Architecture](figures/weatherocc.png)
*Overview of the WeatherOcc3D architecture.*

## 🚀 Highlights
* **Dynamic VLM-Guided Fusion:** Automatically suppresses noise-contaminated sensor channels based on weather and lighting conditions.
* **Plug-and-Play:** Readily integrates into existing baselines (OccMamba, M-CONet) to boost performance.
* **Real-Time Ready:** Adds only **2.14 ms** of latency over standard fusion techniques.

## 📊 Results

Integrating our module yields massive improvements on the **nuScenes-OpenOccupancy** validation set, particularly in challenging environments:

| Condition | OccMamba Baseline | **WeatherOcc3D (Ours)** | Boost |
| :--- | :---: | :---: | :---: |
| **Day** | 26.3 mIoU | **27.1 mIoU** | +0.8 |
| **Night** | 11.8 mIoU | **15.7 mIoU** | +3.9 |
| **Rainy** | 24.1 mIoU | **27.3 mIoU** | +3.2 |

![Qualitative Results](figures/weatherqual.png)
*Qualitative results under adverse weather and lighting conditions using the OccMamba baseline.*

